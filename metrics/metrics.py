import numpy as np
from eval_utils import *
epsilon = 1e-7

def evaluate_distances(PRED, YTRUE, pdb_list, length_dict):
    results_list = []
    for i in range(0, len(PRED[:, 0, 0, 0])):
        L = length_dict[pdb_list[i]]
        D = PRED[i, 0, :L, :L]
        ND = np.copy(YTRUE[i, 0, :L, :L])
        eval_dict = {}
        C = 4.0 / (D + 0.000001)
        C [C > 1.0] = 1.0
        NC = np.copy(ND)
        NC[NC < 8.0] = 1
        NC[NC >= 8.0] = 0
        eval_dict = calc_contact_errors_various_xl(CPRED = C, CTRUE = NC, separation = [12, 24])
        eval_dict.update(calc_dist_errors_various_xl(P = D, Y = ND, L = L))
        for S in [6, 12, 24]:
            for R in [15]:
                eval_dict[f"Cb-LDDT Radius-Cb-LDDT : {R:2d} min-seq-sep: {S:2d}"] = get_LDDT(ND, D, R, S)
        for item in eval_dict:
            print(pdb_list[i], i, len(pdb_list), item, eval_dict[item])
        output = str(pdb_list[i]) + ' ' + str(L) + ' ' + str(i) + ' ' + str(eval_dict)
        output = output.replace(',', '')
        output = output.replace('{', '')
        output = output.replace('}', '')
        output = output.replace(':', '')
        output = output.replace('\'', '')
        results_list.append(output)
    return results_list
