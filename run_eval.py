import os
import argparse
import torch
import numpy as np
import pickle

from models.regressor import SingleInputRegressor
from evaluation.evaluation import evaluate

def main(input_path, shape_label_path, gender_label_path, checkpoint_path, device, silhouettes_from):
    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)

    print("Regressor loaded. Weights from:", checkpoint_path)
    regressor.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    regressor.load_state_dict(checkpoint['best_model_state_dict'])

    shapes = np.load(shape_label_path)
    genders = np.load(gender_label_path)

    image_fnames = [f for f in sorted(os.listdir(input_path)) if f.endswith('.png') or f.endswith('.jpg')]

    file_path = 'evaluation_measurement2.pickle'
    evaluation_file = {
        'pve_neutral': [],
        'height': [],
        'weight': [],
        'chest': [],
        'hip': []
    }
    
    for i in range(len(image_fnames)):
        pve_neutral, weight, height, chest, hip = evaluate(os.path.join(input_path, image_fnames[i]), shapes[i], genders[i], regressor, device, silhouettes_from=silhouettes_from)
        evaluation_file['pve_neutral'].append(pve_neutral[0])
        evaluation_file['weight'].append(weight)
        evaluation_file['height'].append(height)
        evaluation_file['chest'].append(chest)
        evaluation_file['hip'].append(hip)
        print(i)

    with open(file_path, 'wb') as fp:
        pickle.dump(evaluation_file, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to input image/folder of images.')
    parser.add_argument('--shape_label', type=str, help='Path to shape label.')
    parser.add_argument('--gender_label', type=str, help='Path to gender label.')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--silh_from', choices=['densepose', 'pointrend'])
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.input = './ssp-3d/images'
    args.shape_label = './ssp-3d/labels/shapes.npy'
    args.gender_label = './ssp-3d/labels/genders.npy'
    args.checkpoint = 'checkpoints/model_training_past/straps_model_checkpoint_exp001_epoch0.tar'
    args.silh_from = 'densepose'

    main(args.input, args.shape_label, args.gender_label, args.checkpoint, device, args.silh_from)
