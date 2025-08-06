# MAKA-Mmap
This repository implements a deep learning pipeline for predicting real-valued inter-residue distance maps from protein features, along with tools for training, evaluation, and visualization.

📦 Dataset Structure
data_dir/

├── train/

│   ├── features/*.pkl

│   └── distance/*.npy

├── CASP/

│   ├── features/*.pkl

│   └── distance/*.npy
...

📊 Evaluation Only Mode
python eval.py

