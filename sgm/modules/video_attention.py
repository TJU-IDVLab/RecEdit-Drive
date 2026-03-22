import torch

from ..modules.attention import *
from ..modules.diffusionmodules.util import (AlphaBlender, linear,
                                             timestep_embedding)


class TimeMixSequential(nn.Sequential):
    def forward(self, x, context=None, timesteps=None):
        for layer in self:
            x = layer(x, context, timesteps)

        return x


class VideoTransformerBlock(nn.Module):  
    ATTENTION_MODES = {
        "softmax": CrossAttention,
        "softmax-xformers": MemoryEfficientCrossAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        timesteps=None,
        ff_in=False,
        inner_dim=None,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        switch_temporal_ca_to_sa=False,
    ):
        super().__init__()

        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        assert int(n_heads * d_head) == inner_dim

        self.is_res = inner_dim == dim

        if self.ff_in:
            self.norm_in = nn.LayerNorm(dim)
            self.ff_in = FeedForward(
                dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff
            )

        self.timesteps = timesteps
        self.disable_self_attn = disable_self_attn
        if self.disable_self_attn:
            self.attn1 = attn_cls(
                query_dim=inner_dim,
                heads=n_heads,
                dim_head=d_head,
                context_dim=context_dim,
                dropout=dropout,
            )  # is a cross-attention
        else:
            self.attn1 = attn_cls(
                query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
            )  # is a self-attention

        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)

        if disable_temporal_crossattention:
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None
        else:
            self.norm2 = nn.LayerNorm(inner_dim)
            if switch_temporal_ca_to_sa:
                self.attn2 = attn_cls(
                    query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
                )  # is a self-attention
            else:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    context_dim=context_dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                )  # is self-attn if context is none

        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

        self.checkpoint = checkpoint
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x: torch.Tensor, context: torch.Tensor = None, timesteps: int = None
    ) -> torch.Tensor:
        if self.checkpoint:
            return checkpoint(self._forward, x, context, timesteps)
        else:
            return self._forward(x, context, timesteps=timesteps)

    def _forward(self, x, context=None, timesteps=None):
        assert self.timesteps or timesteps
        assert not (self.timesteps and timesteps) or self.timesteps == timesteps
        timesteps = self.timesteps or timesteps
        B, S, C = x.shape
        x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        if self.disable_self_attn:
            x = self.attn1(self.norm1(x), context=context) + x
        else:
            x = self.attn1(self.norm1(x)) + x

        if self.attn2 is not None:
            if self.switch_temporal_ca_to_sa:
                x = self.attn2(self.norm2(x)) + x
            else:
                x = self.attn2(self.norm2(x), context=context) + x
        x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        x = rearrange(
            x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
        )
        return x

    def get_last_layer(self):
        return self.ff.net[-1].weight


class SpatialVideoTransformer(SpatialTransformer): 
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        timesteps=None,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        time_context_dim=None,
        ff_in=False,
        checkpoint=False,
        time_depth=1,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        max_time_embed_period: int = 10000,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            attn_type=attn_mode,
            use_checkpoint=checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
        )
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period

        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim

        self.time_stack = nn.ModuleList(
            [
                VideoTransformerBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=time_context_dim,
                    timesteps=timesteps,
                    checkpoint=checkpoint,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    attn_mode=attn_mode,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                )
                for _ in range(self.depth)
            ]
        )

        assert len(self.time_stack) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = self.in_channels * 4
        self.time_pos_embed = nn.Sequential(
            linear(self.in_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, self.in_channels),
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor, merge_strategy=merge_strategy
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, h, w = x.shape
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context

        if self.use_spatial_context:
            assert (
                context.ndim == 3
            ), f"n dims of spatial context should be 3 but are {context.ndim}"

            time_context = context
            time_context_first_timestep = time_context[::timesteps]
            time_context = repeat(
                time_context_first_timestep, "b ... -> (b n) ...", n=h * w
            )
        elif time_context is not None and not self.use_spatial_context:
            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            if time_context.ndim == 2:
                time_context = rearrange(time_context, "b c -> b 1 c")

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.use_linear:
            x = self.proj_in(x)

        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, "b t -> (b t)")
        t_emb = timestep_embedding(
            num_frames,
            self.in_channels,
            repeat_only=False,
            max_period=self.max_time_embed_period,
        )
        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        for it_, (block, mix_block) in enumerate(
            zip(self.transformer_blocks, self.time_stack)
        ):
            x = block(
                x,
                context=spatial_context,
            )

            x_mix = x
            x_mix = x_mix + emb

            x_mix = mix_block(x_mix, context=time_context, timesteps=timesteps)
            x = self.time_mixer(
                x_spatial=x,
                x_temporal=x_mix,
                image_only_indicator=image_only_indicator,
            )
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out

class Interpolation_CrossAttention(CrossAttention):

    def __init__(
        self,
        query_dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        attn_mode="softmax",
        interpolation_inputs=None,
        down_level=None,
    ):
        super().__init__(
            query_dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            backend=None,
        )
        self.heads = heads
        self.dim_head = dim_head

    def forward(
        self,
        x,
        emb,
        interpolation_inputs,
        context=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
        training_flag=False,
    ):
        c, h, w = x.shape[1:]

        interpolation_dge = interpolation_inputs['interpolation_dge']
        interpolation_indices = interpolation_inputs['interpolation_indices']
        q = torch.repeat_interleave(x, repeats=2, dim=0)
        q_in = rearrange(q, 'b c h w -> b (h w) c')
        if training_flag:
            indice = [[index, item.item()] for index in range(len(interpolation_indices)) for sublist in interpolation_indices[index] for item in sublist]
            kv = [emb[i][j] for i, j in indice]
        else:
            indice = [item.item() for index in range(len(interpolation_indices)) for sublist in interpolation_indices[index] for item in sublist]
            kv = emb[indice]
        kv_in = rearrange(kv, 'b c h w -> b (h w) c')

        output = CrossAttention.forward(
            self,
            x=q_in,
            context=kv_in
        )

        output = rearrange(output, 'b (h w) c -> b c h w', h=h, w=w)
        total_dge = torch.stack([torch.sum(torch.stack(sublist)) for sublist in interpolation_dge])
        total_dge = torch.repeat_interleave(total_dge, repeats=2, dim=0)
        output_dges = torch.stack([item for index in range(len(interpolation_dge)) for sublist in interpolation_dge[index] for item in sublist])
        output_weights = output_dges / total_dge
        output_weights = 1 - output_weights
        output = output * output_weights.view(-1, 1, 1, 1)
        final_output = output[0::2] + output[1::2]

        torch.cuda.empty_cache()
        
        return final_output
    

class Interpolation_MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, activation=nn.GELU):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4  
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        x = self.mlp(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


class CrossFrameAttention(CrossAttention):
    def __init__(
        self,
        query_dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        attn_mode="softmax",
        interpolation_inputs=None,
        down_level=None,
    ):
        super().__init__(
            query_dim,
            context_dim=query_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            backend=None,
        )
        self.heads = heads
        self.dim_head = dim_head

    def smooth_mask(self, mask, sigma=2):
        ksize = int(6*sigma+1)
        if ksize % 2 == 0:
            ksize += 1
        x = torch.arange(ksize) - ksize // 2
        gauss = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss = gauss / gauss.sum()
        gauss = gauss.to(mask.device, mask.dtype)
        
        mask_h = F.conv2d(mask, gauss.view(1,1,1,ksize), padding=(0,ksize//2))
        mask_smooth = F.conv2d(mask_h, gauss.view(1,1,ksize,1), padding=(ksize//2,0))
        
        mask_smooth = (mask_smooth - mask_smooth.min()) / (mask_smooth.max() - mask_smooth.min() + 1e-8)
        return mask_smooth

    def forward(
        self,
        x: torch.Tensor,
        m_ca: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        win_len=3,
        method='gaussian'
    ) -> torch.Tensor:
        torch.cuda.empty_cache()
        b, c, h, w = x.shape
        mask = []

        if m_ca is None:
            m_ca = torch.zeros(b, 1, h, w)

        x = rearrange(x, "b c h w -> b (h w) c")

        q_in = x
        kv_index = [[] for _ in range(win_len)]
        index_list = [0] * (win_len // 2) + [i for i in range(b)] + [b-1] * (win_len // 2)
        for i in range(win_len//2, win_len//2+b):
            for j in range(win_len):
                kv_index[j].append(index_list[i-(win_len//2)+j])

        if method == 'gaussian':
            mask_smooth = self.smooth_mask(m_ca)
            mask_flat = mask_smooth.view(b, 1, h*w)            
            attn_mask = mask_flat.transpose(1,2) @ mask_flat  
            attn_bias = (1.0 - attn_mask) * -1e4
        else:
            mask_smooth = m_ca
            mask_flat = mask_smooth.view(b, 1, h*w)           
            attn_mask = mask_flat.transpose(1,2) @ mask_flat  
            attn_bias = (1.0 - attn_mask) * -1e4


        final_out = torch.zeros_like(q_in)
        for j in range(win_len):
            torch.cuda.empty_cache()
            x_out = CrossAttention.forward(
                self,
                x=q_in,
                mask=attn_bias[kv_index[j]].repeat_interleave(self.heads, dim=0),
                context=x[kv_index[j]],
            )
            final_out = final_out + x_out * torch.tensor(1.0/win_len).to(x.device)

        final_out = rearrange(final_out, "b (h w) c -> b c h w", h=h, w=w)

        return final_out