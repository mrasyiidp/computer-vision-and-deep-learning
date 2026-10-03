import os
import pandas as pd

# Path direktori dataset
dataset_dir = 'dataset_raw'
data = []

# Pindai folder dataset_raw
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(root, file).replace("\\", "/")
            label = 'korban'  # Label tunggal
            data.append({'filename': path, 'label': label})

# Simpan ke metadata.csv
df = pd.DataFrame(data)
df.to_csv('metadata.csv', index=False)
print(f"Berhasil membuat metadata.csv dengan total {len(df)} gambar korban!")