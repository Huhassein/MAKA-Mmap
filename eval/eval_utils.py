import argparse
import sys
import numpy as np
import re
import os
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_score
from scipy.stats import pearsonr

def calc_dist_errors(P, Y, L, dist_thres=None, min_sep=None, top_l_by_x=None, pred_limit=None):
    if Y is None:
        print('ERROR! Y is None!')
        return
    if P is None:
        print('ERROR! P is None!')
        return
    if np.isnan(Y).all():
        print('ERROR! Y is all NaNs!')
        return
    if np.isnan(P).all():
        print('ERROR! P is all NaNs!')
        return
    errors = {}
    errors['mae'] = np.nan
    errors['mse'] = np.nan
    errors['rmse'] = np.nan
    errors['pearsonr'] = np.nan
    errors['count'] = np.nan
    pred_dict = {}
    true_dict = {}
    for p in range(len(Y)):
        for q in range(len(Y)):
            if q - p < min_sep: continue
            if np.isnan(P[p, q]): continue
            if np.isnan(Y[p, q]): continue
            if Y[p, q] >= dist_thres: continue
            if P[p, q] >= pred_limit: continue
            pred_dict[(p, q)] = P[p, q]
            true_dict[(p, q)] = Y[p, q]
    xl = round(L / top_l_by_x)
    pred_list = []
    true_list = []
    for pair in sorted(pred_dict.items(), key=lambda x: x[1]):
        if pair[0] not in true_dict: continue
        pred_list.append(pred_dict[pair[0]])
        true_list.append(true_dict[pair[0]])
        xl -= 1
        if xl == 0: break
    if len(pred_list) > 1:
        pred_list_np = np.array(pred_list)
        true_list_np = np.array(true_list)
        errors['mae'] = round(mean_absolute_error(true_list_np, pred_list_np), 4)
        errors['mse'] = round(mean_squared_error(true_list_np, pred_list_np), 4)
        errors['rmse'] = round(sqrt(errors['mse']), 4)
        errors['pearsonr'] = round(pearsonr(true_list_np, pred_list_np)[0], 4)
        errors['count'] = len(pred_list)
    return errors

def calc_dist_errors_various_xl(P, Y, L, separation=[12, 24]):
    all_metrics = {}
    dist_thres = ['1000']
    topxl = {5: 'Top-L/5', 2: 'Top-L/2', 1: 'Top-L  ', 0.000001: 'ALL    '}
    pred_cutoffs = [15.0]
    for pt in pred_cutoffs:
        for dt in dist_thres:
            for sep in separation:
                for xl in topxl.keys():
                    results = calc_dist_errors(P=P, Y=Y, L=L, dist_thres=int(
                        dt), min_sep=int(sep), top_l_by_x=xl, pred_limit=pt)
                    if len(dist_thres) > 1:
                        all_metrics["prediction-cut-off:" + str(
                            pt) + " native-thres:" + dt + " min-seq-sep:" + str(sep) + " xL:" + topxl[xl]] = results
                    else:
                        all_metrics["prediction-cut-off: " + str(
                            pt) + " min-seq-sep:" + str(sep) + " xL:" + topxl[xl]] = results
    return all_metrics

def calculate_contact_precision(CPRED, CTRUE, minsep, topxl, LPDB=None):
    errors = {}
    errors['precision'] = np.nan
    errors['count'] = np.nan
    L = len(CPRED)
    if LPDB is None: LPDB = len(np.where(~np.isnan(np.diagonal(CTRUE)))[0])
    num_true = 0
    for j in range(0, L):
        for k in range(j, L):
            try:
                CTRUE[j, k]
            except IndexError:
                continue
            if np.isnan(CTRUE[j, k]): continue
            if abs(j - k) < minsep: continue
            if CTRUE[j, k] > 1.0 or CTRUE[j, k] < 0.0: print(
                "WARNING!! True contact at "+str(j)+" "+str(k)+" is "+str(CTRUE[j, k]))
            num_true += 1
    num_pred = 0
    for j in range(0, L):
        for k in range(j, L):
            if np.isnan(CPRED[j, k]): continue
            if abs(j - k) < minsep: continue
            if CPRED[j, k] > 1.0 or CPRED[j, k] < 0.0: print(
                "WARNING!! Predicted probability at "+str(j)+" "+str(k)+" is "+str(CPRED[j, k]))
            num_pred += 1
    if num_true < 1:
        return errors
    p_dict = {}
    for j in range(0, L):
        for k in range(j, L):
            try:
                CTRUE[j, k]
            except IndexError:
                continue
            if np.isnan(CTRUE[j, k]): continue
            if np.isnan(CPRED[j, k]): continue
            if abs(j - k) < minsep: continue
            p_dict[(j, k)] = CPRED[j, k]
    nc_count = 0
    for j in range(0, L):
        for k in range(j, L):
            try:
                CTRUE[j, k]
            except IndexError:
                continue
            if np.isnan(CTRUE[j, k]): continue
            if abs(j - k) < minsep: continue
            if CTRUE[j, k] != 1: continue
            nc_count += 1
    if nc_count < 1:
        return errors
    xl = nc_count
    if topxl == 'L/5': xl = round(0.2 * LPDB)
    if topxl == 'L/2': xl = round(0.5 * LPDB)
    if topxl == 'L': xl = LPDB
    pred_list = []
    true_list = []
    for pair in reversed(sorted(p_dict.items(), key=lambda x: x[1])):
        if np.isnan(CTRUE[pair[0][0], pair[0][0]]): continue
        pred_list.append(1)  # This is assumed to be a +ve prediction
        true_list.append(CTRUE[pair[0][0], pair[0][1]])
        xl -= 1
        if xl == 0: break
    errors['precision'] = round(precision_score(true_list, pred_list), 5)
    errors['count'] = len(true_list)
    return errors

def get_flattened(dmap):
  if dmap.ndim == 1:
    return dmap
  elif dmap.ndim == 2:
    return dmap[np.triu_indices_from(dmap, k=1)]
  else:
    assert False, "ERROR: the passes array has dimension not equal to 2 or 1!"

def get_separations(dmap):
  t_indices = np.triu_indices_from(dmap, k=1)
  separations = np.abs(t_indices[0] - t_indices[1])
  return separations

def get_sep_thresh_b_indices(dmap, thresh, comparator):
  assert comparator in {'gt', 'lt', 'ge',
      'le'}, "ERROR: Unknown comparator for thresholding!"
  dmap_flat = get_flattened(dmap)
  separations = get_separations(dmap)
  if comparator == 'gt':
    threshed = separations > thresh
  elif comparator == 'lt':
    threshed = separations < thresh
  elif comparator == 'ge':
    threshed = separations >= thresh
  elif comparator == 'le':
    threshed = separations <= thresh
  return threshed

def get_dist_thresh_b_indices(dmap, thresh, comparator):
  assert comparator in {'gt', 'lt', 'ge',
      'le'}, "ERROR: Unknown comparator for thresholding!"
  dmap_flat = get_flattened(dmap)
  if comparator == 'gt':
    threshed = dmap_flat > thresh
  elif comparator == 'lt':
    threshed = dmap_flat < thresh
  elif comparator == 'ge':
    threshed = dmap_flat >= thresh
  elif comparator == 'le':
    threshed = dmap_flat <= thresh
  return threshed

def get_LDDT(true_map, pred_map, R=15, sep_thresh=-1, T_set=[0.5, 1, 2, 4], precision=4):
    def get_n_preserved(ref_flat, mod_flat, thresh):
        err = np.abs(ref_flat - mod_flat)
        n_preserved = (err < thresh).sum()
        return n_preserved
    true_flat_map = get_flattened(true_map)
    pred_flat_map = get_flattened(pred_map)
    S_thresh_indices = get_sep_thresh_b_indices(true_map, sep_thresh, 'gt')
    R_thresh_indices = get_dist_thresh_b_indices(true_flat_map, R, 'lt')
    L_indices = S_thresh_indices & R_thresh_indices
    true_flat_in_L = true_flat_map[L_indices]
    pred_flat_in_L = pred_flat_map[L_indices]
    L_n = L_indices.sum()
    preserved_fractions = []
    for _thresh in T_set:
        _n_preserved = get_n_preserved(true_flat_in_L, pred_flat_in_L, _thresh)
        _f_preserved = _n_preserved / L_n
        preserved_fractions.append(_f_preserved)
    lDDT = np.mean(preserved_fractions)
    if precision > 0:
        lDDT = round(lDDT, precision)
    return lDDT
    
def calc_contact_errors_various_xl(CPRED, CTRUE, separation=[12, 24]):
    all_metrics = {}
    topxl = ['L/5', 'L/2', 'L']
    for sep in separation:
        for xl in topxl:
            results = calculate_contact_precision(
                CPRED=CPRED, CTRUE=CTRUE, minsep=sep, topxl=xl)
            all_metrics[f"min-sep:{sep} top-{xl}"] = results
    return all_metrics
