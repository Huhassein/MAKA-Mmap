# MAKA-Mmap
This repository implements a deep learning pipeline for predicting real-valued inter-residue distance maps from protein features, along with tools for training, evaluation, and visualization.

🚀 Training the Model
python train.py \
  -w ./checkpoints/best_model.pkl \
  -n 9000 \
  -e 30 \
  -o ./results_dir \
  -p /path/to/data_dir \
  -v 0

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
python eval.py \
  -w ./checkpoints/best_model.pkl \
  -n 0 \
  -e 0 \
  -o ./results_eval \
  -p /path/to/data_dir \
  -v 1

Predictions will be saved as .npy matrices in the results/ folder.
