import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
npy_path = 'name.npy'
output_path = 'name.png'
matrix = np.load(npy_path)
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, cmap='Spectral', square=True, xticklabels=False, yticklabels=False)
plt.title('Predicted Distance Map')
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()
