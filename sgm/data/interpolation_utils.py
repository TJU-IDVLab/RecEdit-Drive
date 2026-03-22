import pickle
import numpy as np
import cv2
import gzip
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import math
def get_matrix(box):
    elevations, azimuths = None, None
    if box is not None:
        sensor_rotation_object = box.orientation.rotation_matrix.T
        sensor_translation_object = -np.dot(sensor_rotation_object, box.center)

        elevations = np.arcsin(sensor_translation_object[2] / np.linalg.norm(sensor_translation_object))
        azimuths = np.arctan2(sensor_translation_object[1], sensor_translation_object[0])
        elevations = np.pi / 2 - elevations
        azimuths = np.pi + azimuths

    return elevations, azimuths

def min_dis_index(azimuths_rad, ref_num):

    azimuths_deg = np.linspace(0, 360, 21 + 1)[1:] % 360
    azimuths_deg_rad_to_deg = np.degrees(azimuths_rad)

    raw_diff = np.abs(azimuths_deg_rad_to_deg[:, None] - azimuths_deg[None, :])
    diff = np.minimum(raw_diff, 360 - raw_diff)

    min_indices = np.argsort(diff, axis=1)[:, :ref_num]
    diff_azimuths_deg = azimuths_deg_rad_to_deg[:, None] - azimuths_deg[min_indices]
    diff_azimuths_deg = np.where(diff_azimuths_deg > 180, diff_azimuths_deg - 360, diff_azimuths_deg)
    diff_azimuths_deg = np.where(diff_azimuths_deg < -180, diff_azimuths_deg + 360, diff_azimuths_deg)

    return min_indices, diff_azimuths_deg


def dlt_homography(pts_src, pts_dst):
    assert pts_src.shape == pts_dst.shape
    N = pts_src.shape[0]
    assert N == 4, "Four corresponding points must be provided."
    A = []
    for i in range(N):
        x, y = pts_src[i]
        xp, yp = pts_dst[i]
        A.append([x, y, 1, 0, 0, 0, -x*xp, -y*xp, -xp])
        A.append([0, 0, 0, x, y, 1, -x*yp, -y*yp, -yp])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3,3)
    
    if abs(H[2,2]) > 1e-12:
        H = H / H[2,2]
    else:
        H = H / np.linalg.norm(H)
    return H

def get_source_pts(source_box_corners_2d, shape=(900, 1600)):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [source_box_corners_2d.astype(np.int32)], 1)
    ys, xs = np.where(mask == 1)
    points_inside = np.stack([xs, ys], axis=-1)

    return points_inside

def apply_homography(H, pts):
    N = pts.shape[0]
    homo_pts = np.concatenate([pts, np.ones((N, 1))], axis=1).T  
    mapped_pts = (H @ homo_pts).T  
    mapped_pts = mapped_pts[:, :2] / mapped_pts[:, 2:3]  
    return mapped_pts


def warp_image(im, source_pts, target_pts, add_im=None):
    if add_im is None:
        warped_im = np.ones_like(im, dtype=np.float32) * 255
    else:
        warped_im = add_im
    warped_im[target_pts[:, 1], target_pts[:, 0]] = im[source_pts[:, 1], source_pts[:, 0]]
    return warped_im


def get_visible_faces(box):

    CUBE_FACES = [
        [0, 1, 3, 2],  # front
        [4, 7, 5, 6],  # back
        [0, 3, 4, 7],  # right-side
        [1, 5, 6, 2],  # left-side
        [0, 4, 1, 5],  # top
        [2, 6, 3, 7],  # bottom
    ]

    corners = box.corners().T  
    sensor_rotation_object = box.orientation.rotation_matrix.T
    sensor_translation_object = -np.dot(sensor_rotation_object, box.center)
    sensor_corners = np.array([0,0,0])

    visible_faces = []

    for index, face in enumerate(CUBE_FACES):
        pts = corners[face]  # (4,3)
        v1, v2 = pts[1] - pts[0], pts[2] - pts[0]
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        center = pts.mean(axis=0)
        view_dir = sensor_corners - center
        view_dir /= np.linalg.norm(view_dir)
        
        if np.dot(normal, view_dir) <= 1e-7:
            continue

        visible_faces.append(index)

    return visible_faces


def box_rotate_mapping(target_box, source_boxes, camera_intrinsic, im_shape):
    CUBE_FACES = [
        [0, 1, 2, 3],  # front
        [4, 5, 6, 7],  # back
        [0, 4, 7, 3],  # side
        [1, 5, 6, 2],  # side
        [0, 1, 5, 4],  # top
        [2, 3, 7, 6],  # bottom
    ]

    mask_down_shape = [[72, 128], [36, 64], [18, 32], [9, 16]]
    mask_down = [[12.5, 12.5], [25, 25], [50, 50], [100, 100]]

    target_box_corners_2d = view_points(target_box.corners(), camera_intrinsic, normalize=True)[:2, :].T

    source_boxes_corners_2d = []
    source_visible_faces = []
    for source_box in source_boxes:
        box_corners_2d = view_points(source_box.corners(), camera_intrinsic, normalize=True)[:2, :].T
        source_boxes_corners_2d.append(box_corners_2d)
        visible_faces = get_visible_faces(source_box)
        source_visible_faces.append(visible_faces)

    total_down_mapping_pts = []
    total_down_mask_weight = []
    for source_index, source_box_corners_2d in enumerate(source_boxes_corners_2d):
        sub_down_mapping_pts = []
        sub_down_mask_weight = []
        for down in mask_down:
            down_mapping_pts = []
            down_mask_weight = np.zeros((math.ceil(im_shape[0] / down[0]), math.ceil(im_shape[1] / down[1])))
            for face in [CUBE_FACES[i] for i in source_visible_faces[source_index]]:
                target_box_corners_2d_face_down = target_box_corners_2d[face] / down

                H = dlt_homography(source_box_corners_2d[face] / down, target_box_corners_2d_face_down)

                x_min = np.amin(target_box_corners_2d_face_down[:, 1])
                x_max = np.amax(target_box_corners_2d_face_down[:, 1])
                y_min = np.amin(target_box_corners_2d_face_down[:, 0])
                y_max = np.amax(target_box_corners_2d_face_down[:, 0])

                source_pts = get_source_pts(source_box_corners_2d[face] / down)

                mapping_pts = apply_homography(H, source_pts).astype(np.int32)

                mapping_pts[:, [0, 1]] = mapping_pts[:, [1, 0]]
                source_pts[:, [0, 1]] = source_pts[:, [1, 0]]
        
                source_condition = (source_pts[:, 0] < 0) | (source_pts[:, 0] > (im_shape[0] / down[0] - 1)) | \
                            (source_pts[:, 1] < 0) | (source_pts[:, 1] > (im_shape[1] / down[1] - 1))
                target_condition = (mapping_pts[:, 0] < 0) | (mapping_pts[:, 0] > (im_shape[0] / down[0] - 1)) | \
                            (mapping_pts[:, 1] < 0) | (mapping_pts[:, 1] > (im_shape[1] / down[1] - 1))  
                indices_to_delete = np.where(target_condition | source_condition)[0]
                mask = np.ones(mapping_pts.shape[0], dtype=bool)
                mask[indices_to_delete] = False

                filtered_pts = np.concatenate((source_pts, mapping_pts), axis=1)[mask]
                down_mapping_pts.append(filtered_pts)
                sub_mask_weight = np.zeros((math.ceil(im_shape[0] / down[0]), math.ceil(im_shape[1] / down[1])))
                sub_mask_weight[filtered_pts[:, 2], filtered_pts[:, 3]] = 1

                down_mask_weight += sub_mask_weight

            down_mask_weight = np.where(down_mask_weight == 0, np.inf, down_mask_weight)
            down_mask_weight = 1 / down_mask_weight
            sub_down_mask_weight.append(down_mask_weight)
            sub_down_mapping_pts.append(down_mapping_pts)
        total_down_mask_weight.append(sub_down_mask_weight)
        total_down_mapping_pts.append(sub_down_mapping_pts)

    return total_down_mapping_pts, total_down_mask_weight


def get_box_hw(box, camera_intrinsic):

    box_corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :].T
    x_min, y_min = box_corners.min(axis=0)
    x_max, y_max = box_corners.max(axis=0)
    center_norm = np.array([(y_min + y_max) / 2, (x_min + x_max) / 2]) / np.array([900, 1600])

    return y_max - y_min, x_max - x_min, torch.tensor(center_norm, dtype=torch.float32)

def rotate_box(item, target_azimuths_list, ref_num=2):

    keyframe_idx = item['keyframe_idx']
    keyframe_box = item['boxes'][keyframe_idx]
    keyframe_camera_intrinsic = item['camera_intrinsic']
    
    keyframe_box_h, _, keyframe_yx = get_box_hw(keyframe_box, keyframe_camera_intrinsic)

    _, keyframe_box_azimuths = get_matrix(keyframe_box)
    sub_azimuths = target_azimuths_list - keyframe_box_azimuths

    min_indices, diff_azimuths_deg = min_dis_index(np.where(sub_azimuths < 0, sub_azimuths + 2 * np.pi, sub_azimuths), ref_num)
    

    total_warp_mapping = []
    total_warp_mapping_indices = []
    total_warp_mapping_dge = []
    total_warp_mapping_weight = []
    total_warp_ratio = []
    total_warp_posyx = []
    for index in range(len(target_azimuths_list)):
        source_box = []
        warp_mapping_indices = []
        warp_mapping_dge = []
        sub_ratio = []
        sub_posyx = []
        for view_index in range(ref_num):
            view_box = keyframe_box.copy()
            view_quat = Quaternion(axis=[0, 0, 1], angle=diff_azimuths_deg[index][view_index])
            view_box.orientation = view_box.orientation * view_quat

            view_h, _, view_yx = get_box_hw(view_box, keyframe_camera_intrinsic)
            source_box.append(view_box)
            warp_mapping_indices.append(min_indices[index][view_index])
            warp_mapping_dge.append(abs(diff_azimuths_deg[index][view_index]))
            sub_ratio.append(view_h/keyframe_box_h)
            sub_posyx.append(view_yx)
        warp_mapping, warp_mapping_weight = box_rotate_mapping(
            keyframe_box, 
            source_box,
            keyframe_camera_intrinsic, 
            (900, 1600)
        )
        total_warp_mapping.append(warp_mapping)
        total_warp_mapping_indices.append(warp_mapping_indices)
        total_warp_mapping_dge.append(warp_mapping_dge)
        total_warp_mapping_weight.append(warp_mapping_weight)
        total_warp_ratio.append(sub_ratio)
        total_warp_posyx.append(sub_posyx)

    interpolation_inputs = {
        # 'interpolation_flag': , 
        'interpolation_mapping': total_warp_mapping, 
        'interpolation_indices': total_warp_mapping_indices, 
        'interpolation_dge': total_warp_mapping_dge, 
        'interpolation_weight': total_warp_mapping_weight, 
        'interpolation_ratio': total_warp_ratio,
        'interpolation_posyx': total_warp_posyx,
        'interpolation_ref_num': ref_num,
    }

    return interpolation_inputs

