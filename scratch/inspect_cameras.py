import json
import numpy as np

with open('camera comparisions/metadata/camera_database_inventory.json', 'r') as f:
    inv = json.load(f)

for c in inv['cameras']:
    print(f"{c['camera_id']:2d}: {c['camera_name']} ({c['slug']})")

mats = np.load('camera comparisions/sensitivity_functions/camera_sensitivity_matrices.npy')
print("Sensitivity matrices shape:", mats.shape)
