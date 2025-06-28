import os
import numpy as np
import pickle
import subprocess
import sys
import re
from collections import OrderedDict

fasta_dir = "/home/fasta"
output_dir = "/home/pkl"
aln_feature_dir = "/home/dataset/features/"
pssm_dir = "/home/datasetfeatures/pssm"

for directory in [output_dir, aln_feature_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_fasta(fasta_path):
    seq_lines = []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('>'):
                seq_lines.append(line)
    return ''.join(seq_lines)

def extract_ccmpred_features(ccmpred_file, seq_length):
    ccmpred = np.zeros((seq_length, seq_length))
    with open(ccmpred_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            ccmpred[i, :] = [float(x) for x in line.strip().split()]
    return ccmpred

def extract_freecontact_features(freecontact_file, seq_length):
    freecon = np.zeros((seq_length, seq_length))
    with open(freecontact_file, 'r') as f:
        for line in f:
            c = line.strip().split()
            if len(c) >= 5:
                i = int(c[0]) - 1
                j = int(c[1]) - 1
                score = float(c[4])
                if i < seq_length and j < seq_length:
                    freecon[i, j] = score
                    freecon[j, i] = score
                else:
                    print(f"Warning: Index out of bounds in {freecontact_file}, i={i}, j={j}")
    print(f"FreeContact feature extracted for {freecontact_file}: Non-zero elements count = {np.count_nonzero(freecon)}")
    return freecon

def extract_entropy_features(colstats_file, seq_length):
    entropy = []
    with open(colstats_file, 'r') as f:
        for line in f.readlines()[4:]:
            fields = line.strip().split()
            entropy.append(float(fields[21]))
    return np.array(entropy, dtype=np.float16)

def extract_potential_features(pairstats_file, seq_length):
    potential = np.zeros((seq_length, seq_length))
    with open(pairstats_file, 'r') as f:
        for line in f.readlines():
            fields = line.strip().split()
            i = int(fields[0]) - 1
            j = int(fields[1]) - 1
            potential[i, j] = float(fields[2])
            potential[j, i] = float(fields[2])
    return potential

fasta_ids = [os.path.splitext(f)[0] for f in os.listdir(fasta_dir) if f.endswith(".fasta")]
aln_ids = [os.path.splitext(f)[0] for f in os.listdir(aln_dir) if f.endswith(".aln")]
protein_ids = set(fasta_ids + aln_ids)

print(f"Total protein IDs: {len(protein_ids)}")

for protein_id in protein_ids:
    print(f"Processing {protein_id}...")

    features = {}
    fasta_path = os.path.join(fasta_dir, f"{protein_id}.fasta")
    if os.path.exists(fasta_path):
        seq = read_fasta(fasta_path)
        features['seq'] = seq
    else:
        features['seq'] = ""

    aln_path = os.path.join(aln_dir, f"{protein_id}.aln")
    if os.path.exists(aln_path):
        ccmpred_file = os.path.join(aln_feature_dir, f"{protein_id}.ccmpred")
        freecontact_file = os.path.join(aln_feature_dir, f"{protein_id}.freecontact.rr")
        colstats_file = os.path.join(aln_feature_dir, f"{protein_id}.colstats")
        pairstats_file = os.path.join(aln_feature_dir, f"{protein_id}.pairstats")
        pssm_file = os.path.join(pssm_dir, f"{protein_id}.npy")

        with open(aln_path, 'r') as f:
            lines = f.readlines()
            seq_length = len(lines[0].strip())

        if os.path.exists(ccmpred_file):
            features['ccmpred'] = extract_ccmpred_features(ccmpred_file, seq_length)
        if os.path.exists(freecontact_file):
            features['freecon'] = extract_freecontact_features(freecontact_file, seq_length)

        if os.path.exists(colstats_file):
            features['entropy'] = extract_entropy_features(colstats_file, seq_length)
        if os.path.exists(pairstats_file):
            features['potential'] = extract_potential_features(pairstats_file, seq_length)

        if os.path.exists(pssm_file):
            pssm = np.load(pssm_file)
            features['pssm'] = pssm

        features['ccmpred'] = np.zeros((seq_length, seq_length))
        features['freecon'] = np.zeros((seq_length, seq_length))
        features['entropy'] = np.zeros(seq_length)
        features['potential'] = np.zeros((seq_length, seq_length))
        features['pssm'] = np.zeros((seq_length, 22))

    ordered_features = OrderedDict([
        ('seq', features['seq']),
        ('ccmpred', features['ccmpred']),
        ('freecon', features['freecon']),
        ('entropy', features['entropy']),
        ('potential', features['potential']),
        ('pssm', features['pssm'])
    ])

    pkl_file = os.path.join(output_dir, f"{protein_id}.pkl")
    with open(pkl_file, 'wb') as f:
        pickle.dump(ordered_features, f)

    print(f"Saved features for {protein_id} to {pkl_file}")
