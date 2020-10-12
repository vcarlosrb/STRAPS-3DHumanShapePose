import torch
from smplx.lbs import batch_rodrigues

def getReposedRotmats(batch_size, device):
    pose_r = []
    for index in range(0,72):
        if (index == 50) | (index == 53):
            if index == 50:
                pose_r.append(5.6)
            if index == 53:
                pose_r.append(-5.6)
        else:
            pose_r.append(0)

    reposed_pose = []
    for index in range(0, batch_size):
        reposed_pose.append(pose_r)
    
    reposed_pose = torch.FloatTensor(reposed_pose).to(device)
    reposed_pose_rotmats = batch_rodrigues(reposed_pose[:, 3:].contiguous().view(-1, 3))
    reposed_pose_rotmats = reposed_pose_rotmats.view(-1, 23, 3, 3)
    reposed_glob_rotmats = batch_rodrigues(reposed_pose[:, :3].contiguous().view(-1, 3))
    reposed_glob_rotmats = reposed_glob_rotmats.unsqueeze(1)

    return reposed_pose_rotmats, reposed_glob_rotmats