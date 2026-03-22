from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import repeat, rearrange

from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...modules.encoders.modules import GeneralConditioner
from ...util import append_dims, instantiate_from_config
from .denoiser import Denoiser


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
        reweight: int = 1.0,
    ):
        super().__init__()

        assert loss_type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level

        if loss_type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)
        self.reweight = reweight

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        conditioner_3d: GeneralConditioner,
        input: torch.Tensor,
        input_3d: torch.Tensor,
        batch: Dict,
        copy_bg: float = 0.0,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        cond_3d = conditioner_3d(batch)
        for k, v in cond.items():
            if isinstance(v, torch.Tensor) and v.shape[0] != batch['num_video_frames']:
                v = repeat(v, "b ... -> b t ...", t=batch['num_video_frames'])
                v = rearrange(v, "b t ... -> (b t) ...", t=batch['num_video_frames'])
                cond[k] = v
        for k, v in cond_3d.items():
            if isinstance(v, torch.Tensor) and v.shape[0] != batch['num_video_frames_3d']:
                v = repeat(v, "b ... -> b t ...", t=batch['num_video_frames_3d'])
                v = rearrange(v, "b t ... -> (b t) ...", t=batch['num_video_frames_3d'])
                cond_3d[k] = v
        if copy_bg == 0:
            return self._forward(network, denoiser, cond, cond_3d, input, input_3d, batch)
        else:
            return self._forward(network, denoiser, cond, cond_3d, input, input_3d, batch, copy_bg)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        cond_3d: Dict,
        input: torch.Tensor,
        input_3d: torch.Tensor,
        batch: Dict,
        copy_bg: float = 0.0
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        noise = torch.randn_like(input)
        
        if batch['sv3d_mode'] == 'interpolation':
            noise_3d = torch.randn((input.shape[0],) + input_3d.shape, device=input_3d.device)
            input_3d = input_3d.unsqueeze(0).expand(input.shape[0], -1, -1, -1, -1)

            sigmas = self.sigma_sampler(n_samples=input.shape[0]).to(input)
            sigmas_3d = sigmas.unsqueeze(1).expand(-1, 21)
        elif batch['sv3d_mode'] == 'driveeditor' or batch['sv3d_mode'] == 'sv3d':
            noise_3d = torch.randn_like(input_3d)
            sigmas_3d = self.sigma_sampler(input_3d.shape[0]).to(input)
            sigmas = sigmas_3d[batch['indices_3d']]  # This may affect training performance

        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        sigmas_bc_3d = append_dims(sigmas_3d, input_3d.ndim)
        noised_input = self.get_noised_input(sigmas_bc, noise, input)   
        noised_input_3d = self.get_noised_input(sigmas_bc_3d, noise_3d, input_3d)   

        model_output, _, e_sv3d_time2, e_x_time2 = denoiser(
            network, noised_input, noised_input_3d, sigmas, sigmas_3d, cond, cond_3d, \
                sv3d_mode = batch['sv3d_mode'], interpolation_inputs = batch.get('interpolation_inputs', None), training_flag=True, **additional_model_inputs
        )
        w = append_dims(self.loss_weighting(sigmas), input.ndim)

        if copy_bg == 0:
            return self.get_loss(model_output, input, w, additional_model_inputs['mask_fuse'])
        else:
            # 判断哪些部分只计算前景的loss
            copy_bg_mask = append_dims((sigmas > 12).int(), input.ndim)
            bg_mask = (additional_model_inputs['mask_fuse'].repeat(1, 4, 1, 1) + 1 ) / 2
            new_model_output = model_output * (1 - copy_bg_mask) \
                            + model_output * bg_mask * copy_bg_mask \
                            + input * (1 - bg_mask) * copy_bg_mask
            return self.get_loss(new_model_output, input, w, additional_model_inputs['mask_fuse'], copy_bg_mask)

    def get_loss(self, model_output, target, w, mask, copy_bg_mask=None):
        mask = (mask.repeat(1, 4, 1, 1) + 1 ) / 2 * (self.reweight - 1) + 1
        mask = mask / torch.mean(mask, dim=(2, 3), keepdim=True)
        if self.loss_type == "l2":
            return torch.mean(
                (mask * w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
