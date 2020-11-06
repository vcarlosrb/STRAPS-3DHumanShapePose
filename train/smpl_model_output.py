import torch
import numpy as np

class SMPLOutput(object):
    vertices = []
    joints = []

def getSMPLModelOutputByBtach(smpl_mode_f, smpl_mode_m, pose_rotmats, glob_rotmats, shape, gender, pose2rot, device):
    vertices = []
    joints = []

    for index in range(len(gender)):
        n_shape = torch.unsqueeze(shape[index], 0).to(device)
        n_pose_rotmats = torch.unsqueeze(pose_rotmats[index], 0).to(device)
        n_glob_rotmats = torch.unsqueeze(glob_rotmats[index], 0).to(device)
        if gender[index] == 'female':
            smpl_output = smpl_mode_f(
                betas=n_shape,
                body_pose=n_pose_rotmats,
                global_orient=n_glob_rotmats,
                pose2rot=pose2rot
            )
        elif gender[index] == 'male':
            smpl_output = smpl_mode_m(
                betas=n_shape,
                body_pose=n_pose_rotmats,
                global_orient=n_glob_rotmats,
                pose2rot=pose2rot
            )
        vertices.append(smpl_output.vertices)
        joints.append(smpl_output.joints)

    vertices = torch.cat(vertices)
    joints = torch.cat(joints)
    
    output = SMPLOutput()
    output.vertices = vertices
    output.joints = joints

    return output