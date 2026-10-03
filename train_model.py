import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.svm import OneClassSVM
from model_utils import extract_features, measure_latency, plot_and_save_distances

# 1. BACA METADATA.CSV
df = pd.read_csv('metadata.csv')
file_paths = df['filename'].tolist()

if len(file_paths) == 0:
    raise ValueError("Tidak ada gambar yang ditemukan di metadata.csv! Jalankan generate_metadata.py terlebih dahulu.")

# 2. FUNGSI PEMBENTUK FEATURE EXTRACTOR (RESNET)
def build_feature_extractor(mode='feature_extraction'):
    # Menggunakan pretrained ResNet50 (Bobot ImageNet)
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    if mode == 'feature_extraction':        # Freeze seluruh backbone ResNet
        base_model.trainable = False
    elif mode == 'partial_fine_tuning':     # Unfreeze 20 layer teratas
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
    elif mode == 'full_fine_tuning':        # Unfreeze semua layer
        base_model.trainable = True

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    return model

# 3. EKSEKUSI PELATIHAN ONE-CLASS PADA 3 MODE
results = []
modes = ['feature_extraction', 'partial_fine_tuning', 'full_fine_tuning']

print("\n==========================================")
print("  MULAI PELATIHAN ONE-CLASS RESNET DATASET")
print("==========================================")

for mode in modes:
    print(f"\n---> Memproses Mode: {mode.upper()}")
    
    # Buat feature extractor berbasis ResNet
    extractor = build_feature_extractor(mode)
    
    # Ekstrak fitur visual dari seluruh gambar korban
    features = extract_features(extractor, file_paths)
    
    # Latih One-Class SVM
    oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    oc_svm.fit(features)
    
    # Hitung rata-rata jarak fitur ke pusat kluster
    center = np.mean(features, axis=0)
    distances = np.linalg.norm(features - center, axis=1)
    avg_distance = np.mean(distances)
    
    # Ukur latensi inferensi (ms/frame)
    latency = measure_latency(extractor, file_paths[0])
    
    # Simpan grafik sebaran jarak
    plot_and_save_distances(distances, mode)
    
    results.append({
        'Mode Experiment': mode,
        'Jumlah Data Korban': len(file_paths),
        'Avg Feature Distance': f"{avg_distance:.4f}",
        'Latency (ms)': f"{latency:.2f} ms"
    })

# 4. TAMPILKAN DAN SIMPAN RINGKASAN HASIL
df_results = pd.DataFrame(results)
print("\n=== TABEL HASIL ANALISIS DATASET KORBAN (RESNET) ===")
print(df_results.to_string(index=False))

df_results.to_csv('results_summary.csv', index=False)
print("\nHasil eksekusi berhasil disimpan ke 'results_summary.csv'")