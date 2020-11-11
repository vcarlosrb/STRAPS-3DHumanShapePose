import os
import cv2
import numpy as np
import torch
import trimesh
from smplx.lbs import batch_rodrigues

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from PointRend.point_rend import add_pointrend_config
from DensePose.densepose import add_densepose_config

import config

from predict.predict_joints2D import predict_joints2D
from predict.predict_silhouette_pointrend import predict_silhouette_pointrend
from predict.predict_densepose import predict_densepose

from models.smpl_official import SMPL
from renderers.weak_perspective_pyrender_renderer import Renderer

from utils.image_utils import pad_to_square
from utils.cam_utils import orthographic_project_torch
from utils.reposed_utils import getReposedRotmats
from utils.joints2d_utils import undo_keypoint_normalisation
from utils.label_conversions import convert_multiclass_to_binary_labels, \
    convert_2Djoints_to_gaussian_heatmaps
from utils.rigid_transform_utils import rot6d_to_rotmat

from bodyMeasurement.body_measurement_from_smpl import getHeight

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

from bodyMeasurement.body_measurement_from_smpl import getBodyMeasurement


def setup_detectron2_predictors(silhouettes_from='densepose'):
    # Keypoint-RCNN
    kprcnn_config_file = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    kprcnn_cfg = get_cfg()
    kprcnn_cfg.merge_from_file(model_zoo.get_config_file(kprcnn_config_file))
    kprcnn_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    kprcnn_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(kprcnn_config_file)
    kprcnn_cfg.freeze()
    joints2D_predictor = DefaultPredictor(kprcnn_cfg)

    if silhouettes_from == 'pointrend':
        # PointRend-RCNN-R50-FPN
        pointrend_config_file = "PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
        pointrend_cfg = get_cfg()
        add_pointrend_config(pointrend_cfg)
        pointrend_cfg.merge_from_file(pointrend_config_file)
        pointrend_cfg.MODEL.WEIGHTS = "checkpoints/pointrend_rcnn_R_50_fpn.pkl"
        pointrend_cfg.freeze()
        silhouette_predictor = DefaultPredictor(pointrend_cfg)
    elif silhouettes_from == 'densepose':
        # DensePose-RCNN-R101-FPN
        densepose_config_file = "DensePose/configs/densepose_rcnn_R_101_FPN_s1x.yaml"
        densepose_cfg = get_cfg()
        add_densepose_config(densepose_cfg)
        densepose_cfg.merge_from_file(densepose_config_file)
        densepose_cfg.MODEL.WEIGHTS = "checkpoints/densepose_rcnn_R_101_fpn_s1x.pkl"
        densepose_cfg.freeze()
        silhouette_predictor = DefaultPredictor(densepose_cfg)

    return joints2D_predictor, silhouette_predictor


def create_proxy_representation(silhouette,
                                joints2D,
                                in_wh,
                                out_wh):
    silhouette = cv2.resize(silhouette, (out_wh, out_wh),
                            interpolation=cv2.INTER_NEAREST)
    joints2D = joints2D[:, :2]
    joints2D = joints2D * np.array([out_wh / float(in_wh),
                                    out_wh / float(in_wh)])
    heatmaps = convert_2Djoints_to_gaussian_heatmaps(joints2D.astype(np.int16),
                                                     out_wh)
    proxy_rep = np.concatenate([silhouette[:, :, None], heatmaps], axis=-1)
    proxy_rep = np.transpose(proxy_rep, [2, 0, 1])  # (C, out_wh, out_WH)

    return proxy_rep


def evaluate(input,
               shape,
               gender,
               regressor,
               device,
               silhouettes_from='densepose',
               proxy_rep_input_wh=512):

    # Set-up proxy representation predictors.
    joints2D_predictor, silhouette_predictor = setup_detectron2_predictors(silhouettes_from=silhouettes_from)

    image = cv2.imread(input)
    # Pre-process for 2D detectors
    image = pad_to_square(image)
    image = cv2.resize(image, (proxy_rep_input_wh, proxy_rep_input_wh),
                        interpolation=cv2.INTER_LINEAR)

    # Predict 2D
    joints2D, joints2D_vis = predict_joints2D(image, joints2D_predictor)
    if silhouettes_from == 'pointrend':
        silhouette, silhouette_vis = predict_silhouette_pointrend(image,
                                                                    silhouette_predictor)
    elif silhouettes_from == 'densepose':
        silhouette, silhouette_vis = predict_densepose(image, silhouette_predictor)
        silhouette = convert_multiclass_to_binary_labels(silhouette)

    # Create proxy representation
    proxy_rep = create_proxy_representation(silhouette, joints2D,
                                            in_wh=proxy_rep_input_wh,
                                            out_wh=config.REGRESSOR_IMG_WH)
    proxy_rep = proxy_rep[None, :, :, :]  # add batch dimension
    proxy_rep = torch.from_numpy(proxy_rep).float().to(device)

    height = getHeightFromSample(torch.from_numpy(shape.reshape(1, shape.shape[0])).to(device).detach(), gender, device)
    height = np.asarray([height])
    height = torch.FloatTensor(height.reshape(height.shape[0], 1)).to(device)

    if gender == 'm':
        gender_n = np.asarray([1])
        gender_n = torch.FloatTensor(gender_n.reshape(gender_n.shape[0], 1)).to(device)
    elif gender == 'f':
        gender_n = np.asarray([0])
        gender_n = torch.FloatTensor(gender_n.reshape(gender_n.shape[0], 1)).to(device)

    # Predict 3D
    regressor.eval()
    with torch.no_grad():
        pred_cam_wp, pred_pose, pred_shape = regressor(proxy_rep, height, gender_n)

    #pve_neutral_pose_scale = compute_pve_neutral_pose_scale_corrected(pred_shape.to(device).detach(), torch.from_numpy(shape.reshape(1, shape.shape[0])).to(device).detach(), gender, device)
    pve_neutral_pose_scale, weight_error, height_error, chest_error, hip_error = measurementError(pred_shape.to(device).detach(), torch.from_numpy(shape.reshape(1, shape.shape[0])).to(device).detach(), gender, device)
        
    return pve_neutral_pose_scale, weight_error, height_error, chest_error, hip_error

def getHeightFromSample(shape, gender, device):
    reposed_pose_rotmats, reposed_glob_rotmats = getReposedRotmats(1, device)
    if gender == 'm':
        smpl_male = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='male').to(device)
        target_smpl_neutral_pose_output = smpl_male(
            betas=shape,
            body_pose=reposed_pose_rotmats,
            global_orient=reposed_glob_rotmats,
            pose2rot=False
        )
        faces = smpl_male.faces
    elif gender == 'f':
        smpl_female = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='female').to(device)
        target_smpl_neutral_pose_output = smpl_female(
            betas=shape,
            body_pose=reposed_pose_rotmats,
            global_orient=reposed_glob_rotmats,
            pose2rot=False
        )
        faces = smpl_female.faces
    
    target_smpl_neutral_pose_vertices = target_smpl_neutral_pose_output.vertices
    vertices = target_smpl_neutral_pose_vertices.cpu().detach().numpy()
    vertices = vertices.reshape(vertices.shape[1], vertices.shape[2])

    mesh = trimesh.Trimesh(vertices, faces)
    return getHeight(mesh)


def measurementError(predicted_smpl_shape, target_smpl_shape, gender, device):
    reposed_pose_rotmats, reposed_glob_rotmats = getReposedRotmats(1, device)
    faces = None
    if gender == 'm':
        smpl_male = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='male').to(device)
        pred_smpl_neutral_pose_output = smpl_male(
            betas=predicted_smpl_shape,
            body_pose=reposed_pose_rotmats,
            global_orient=reposed_glob_rotmats,
            pose2rot=False)
        target_smpl_neutral_pose_output = smpl_male(
            betas=target_smpl_shape,
            body_pose=reposed_pose_rotmats,
            global_orient=reposed_glob_rotmats,
            pose2rot=False
        )
        faces = smpl_male.faces
    elif gender == 'f':
        smpl_female = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='female').to(device)
        pred_smpl_neutral_pose_output = smpl_female(
            betas=predicted_smpl_shape,
            body_pose=reposed_pose_rotmats,
            global_orient=reposed_glob_rotmats,
            pose2rot=False
        )
        target_smpl_neutral_pose_output = smpl_female(
            betas=target_smpl_shape,
            body_pose=reposed_pose_rotmats,
            global_orient=reposed_glob_rotmats,
            pose2rot=False
        )
        faces = smpl_female.faces


    pred_smpl_neutral_pose_vertices = pred_smpl_neutral_pose_output.vertices
    target_smpl_neutral_pose_vertices = target_smpl_neutral_pose_output.vertices

    # Rescale such that RMSD of predicted vertex mesh is the same as RMSD of target mesh.
    # This is done to combat scale vs camera depth ambiguity.
    pred_smpl_neutral_pose_vertices_rescale = scale_and_translation_transform_batch(pred_smpl_neutral_pose_vertices,
                                                                                    target_smpl_neutral_pose_vertices)

    # Compute PVE-T-SC
    pve_neutral_pose_scale_corrected = np.linalg.norm(pred_smpl_neutral_pose_vertices_rescale
                                                      - target_smpl_neutral_pose_vertices.detach().cpu().numpy(),
                                                      axis=-1)  # (1, 6890)

    # Measurements
    weight_pred, height_pred, chest_pred, hip_pred = getBodyMeasurement(pred_smpl_neutral_pose_vertices, faces)
    weight_target, height_target, chest_target, hip_target = getBodyMeasurement(target_smpl_neutral_pose_vertices, faces)

    weight_error = weight_target - weight_pred
    height_error = height_target - height_pred
    chest_error = chest_target - chest_pred
    hip_error = hip_target - hip_pred

    return pve_neutral_pose_scale_corrected, weight_error, height_error, chest_error, hip_error

def compute_pve_neutral_pose_scale_corrected(predicted_smpl_shape, target_smpl_shape, gender, device):
    """
    Given predicted and target SMPL shape parameters, computes neutral-pose per-vertex error
    after scale-correction (to account for scale vs camera depth ambiguity).
    :param predicted_smpl_parameters: predicted SMPL shape parameters tensor with shape (1, 10)
    :param target_smpl_parameters: target SMPL shape parameters tensor with shape (1, 10)
    :param gender: gender of target
    """

    # Get neutral pose vertices
    if gender == 'm':
        smpl_male = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='male').to(device)
        pred_smpl_neutral_pose_output = smpl_male(betas=predicted_smpl_shape)
        target_smpl_neutral_pose_output = smpl_male(betas=target_smpl_shape)
    elif gender == 'f':
        smpl_female = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='female').to(device)
        pred_smpl_neutral_pose_output = smpl_female(betas=predicted_smpl_shape)
        target_smpl_neutral_pose_output = smpl_female(betas=target_smpl_shape)

    pred_smpl_neutral_pose_vertices = pred_smpl_neutral_pose_output.vertices
    target_smpl_neutral_pose_vertices = target_smpl_neutral_pose_output.vertices

    # Rescale such that RMSD of predicted vertex mesh is the same as RMSD of target mesh.
    # This is done to combat scale vs camera depth ambiguity.
    pred_smpl_neutral_pose_vertices_rescale = scale_and_translation_transform_batch(pred_smpl_neutral_pose_vertices,
                                                                                    target_smpl_neutral_pose_vertices)

    # Compute PVE-T-SC
    pve_neutral_pose_scale_corrected = np.linalg.norm(pred_smpl_neutral_pose_vertices_rescale
                                                      - target_smpl_neutral_pose_vertices.detach().cpu().numpy(),
                                                      axis=-1)  # (1, 6890)

    return pve_neutral_pose_scale_corrected


def scale_and_translation_transform_batch(P, T):
    """
    First normalises batch of input 3D meshes P such that each mesh has mean (0, 0, 0) and
    RMS distance from mean = 1.
    Then transforms P such that it has the same mean and RMSD as T.
    :param P: (batch_size, N, 3) batch of N 3D meshes to transform.
    :param T: (batch_size, N, 3) batch of N reference 3D meshes.
    :return: P transformed
    """

    P = P.detach().cpu().numpy()
    T = T.detach().cpu().numpy()

    P_mean = np.mean(P, axis=1, keepdims=True)
    P_trans = P - P_mean
    P_scale = np.sqrt(np.sum(P_trans ** 2, axis=(1, 2), keepdims=True) / P.shape[1])
    P_normalised = P_trans / P_scale

    T_mean = np.mean(T, axis=1, keepdims=True)
    T_scale = np.sqrt(np.sum((T - T_mean) ** 2, axis=(1, 2), keepdims=True) / T.shape[1])

    P_transformed = P_normalised * T_scale + T_mean

    return P_transformed