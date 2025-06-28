import pickle
import random
import os
import numpy as np
import torch

def load_list(file_lst, max_items=1000000):
    if max_items < 0:
        max_items = 1000000
    protein_list = []
    with open(file_lst, 'r') as f:
        for line in f:
            protein_list.append(line.strip().split()[0])
    return protein_list[:max_items]
    
def get_input_output_dist(pdb_id_list, all_feat_paths, all_dist_paths, pad_size, OUTL, expected_n_channels):
    XX = np.full((len(pdb_id_list), OUTL, OUTL, expected_n_channels), 0.0, dtype=np.float32)
    YY = np.full((len(pdb_id_list), OUTL, OUTL, 1), 100.0, dtype=np.float32)
    for i, pdb in enumerate(pdb_id_list):
        X = get_feature(pdb, all_feat_paths, expected_n_channels)
        Y0 = get_map(pdb, all_dist_paths, X.shape[0])
        l = X.shape[0]
        Y = np.full((l, l), np.nan, dtype=np.float32)
        Y[:Y0.shape[0], :Y0.shape[0]] = Y0
        Xpadded = np.zeros((l + pad_size, l + pad_size, X.shape[2]), dtype=np.float32)
        Xpadded[pad_size // 2 : pad_size // 2 + l, pad_size // 2 : pad_size // 2 + l, :] = X
        Ypadded = np.full((l + pad_size, l + pad_size), 100.0, dtype=np.float32)
        Ypadded[pad_size // 2 : pad_size // 2 + l, pad_size // 2 : pad_size // 2 + l] = Y
        l = Xpadded.shape[0]
        if l <= OUTL:
            XX[i, :l, :l, :] = Xpadded
            YY[i, :l, :l, 0] = Ypadded
        else:
            if l < OUTL:
                raise ValueError(f"Input too small: padded length {l} < OUTL {OUTL}")
            rx = random.randint(0, l - OUTL)
            ry = random.randint(0, l - OUTL)
            XX[i] = Xpadded[rx:rx + OUTL, ry:ry + OUTL, :]
            YY[i, :, :, 0] = Ypadded[rx:rx + OUTL, ry:ry + OUTL]
    return torch.tensor(XX), torch.tensor(YY)
                
def get_feature(pdb, all_feat_paths, expected_n_channels):
    features = None
    for path in all_feat_paths:
        fpath = os.path.join(path, pdb + '.pkl')
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                features = pickle.load(f)
            break
    if features is None:
        raise FileNotFoundError(f"Feature file for {pdb} not found in {all_feat_paths}")
    l = len(features['seq'])
    X = np.full((l, l, expected_n_channels), 0.0, dtype=np.float32)
    fi = 0
    for j in range(22):
        a = np.repeat(features['pssm'][:, j].reshape(1, l), l, axis=0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
    a = np.repeat(features['entropy'].reshape(1, l), l, axis=0)
    X[:, :, fi] = a
    fi += 1
    X[:, :, fi] = a.T
    fi += 1
    X[:, :, fi] = features['ccmpred']; fi += 1
    X[:, :, fi] = features['freecon']; fi += 1
    X[:, :, fi] = features['potential']; fi += 1
    assert fi == expected_n_channels
    return X

def get_map(pdb, all_dist_paths, expected_l=-1):
    cb_map = None
    for path in all_dist_paths:
        fpath = os.path.join(path, pdb + '-cb.npy')
        if os.path.exists(fpath):
            try:
                data = np.load(fpath, allow_pickle=True)
                if isinstance(data, np.ndarray) and data.dtype == object and len(data) == 3:
                    _, _, cb_map = data
                elif isinstance(data, np.ndarray) and data.ndim == 2:
                    cb_map = data
                else:
                    raise ValueError(f"Unsupported format in {fpath}")
                break
            except Exception as e:
                print(f"Failed to load {fpath}: {e}")
                continue
    if cb_map is None:
        raise FileNotFoundError(f"Distance map for {pdb} not found in {all_dist_paths}")
    if expected_l > 0 and cb_map.shape[0] != expected_l:
        if cb_map.shape[0] < expected_l:
            pad = expected_l - cb_map.shape[0]
            cb_map = np.pad(cb_map, ((0, pad), (0, pad)), mode='constant', constant_values=1.0)
        else:
            cb_map = cb_map[:expected_l, :expected_l]
    Y = cb_map
    Y[Y < 1.0] = 1.0
    if np.any(np.isnan(np.diagonal(Y))):
        print(f"WARNING: NaN on diagonal in {pdb}")
    if Y.shape[0] >= 2:
        Y[0, 0] = Y[0, 1] if not np.isnan(Y[0, 1]) else 1.0
        Y[-1, -1] = Y[-1, -2] if not np.isnan(Y[-1, -2]) else 1.0
    for q in range(1, Y.shape[0] - 1):
        if np.isnan(Y[q, q]):
            left = Y[q, q - 1]
            right = Y[q, q + 1]
            if np.isnan(left) and np.isnan(right):
                Y[q, q] = 1.0
            elif np.isnan(left):
                Y[q, q] = right
            elif np.isnan(right):
                Y[q, q] = left
            else:
                Y[q, q] = (left + right) / 2.0
    return Y

