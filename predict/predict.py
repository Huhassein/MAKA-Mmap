import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from model import MAKA

pkl_path = '1a0b_1_A.pkl'
model_weights_path = './maka.pth' 
output_npy_path = '1a0b_1_A.npy'
expected_n_channels = 49 

def get_feature(pkl_file, expected_n_channels):
    features = pickle.load(open(pkl_file, 'rb'))
    l = len(features['seq'])
    X = np.full((l, l, expected_n_channels), 0.0)
    fi = 0
    pssm = features['pssm']   # (l, 22)
    for j in range(22):
        a = np.repeat(pssm[:, j].reshape(1, l), l, axis=0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
    entropy = features['entropy']  # (l,)
    a = np.repeat(entropy.reshape(1, l), l, axis=0)
    X[:, :, fi] = a
    fi += 1
    X[:, :, fi] = a.T
    fi += 1
    X[:, :, fi] = features['ccmpred']
    fi += 1
    X[:, :, fi] = features['freecon']
    fi += 1
    X[:, :, fi] = features['potential']
    fi += 1
    assert fi == expected_n_channels
    return X

def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_tensor = get_feature(pkl_path, expected_n_channels)
    feature_tensor = np.transpose(feature_tensor, (2, 0, 1))     
    feature_tensor = torch.tensor(feature_tensor, dtype=torch.float32).unsqueeze(0).to(device)
    model = MAKA(expected_n_channels=expected_n_channels).to(device)
    model.eval()
    state_dict = torch.load(model_weights_path, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if torch.cuda.device_count() > 1 and not k.startswith('module.'):
            name = 'module.' + k
        elif torch.cuda.device_count() <= 1 and k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    with torch.no_grad():
        prediction = model(feature_tensor)
        prediction_np = prediction.cpu().numpy()[0]
    np.save(output_npy_path, prediction_np)

if __name__ == '__main__':
    predict()

