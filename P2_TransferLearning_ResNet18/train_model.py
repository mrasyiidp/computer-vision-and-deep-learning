import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.svm import OneClassSVM
from classification_models.tfkeras import Classifiers
from model_utils import extract_features, measure_latency, plot_and_save_distances

# 1. BACA METADATA
df = pd.read_csv('metadata.csv')
file_paths = df['filename'].tolist()

if len(file_paths) == 0:
    raise ValueError("Tiada gambar dijumpai dalam metadata.csv!")

# 2. LOAD MURNI ARSITEKTUR RESNET-18
ResNet18, _ = Classifiers.get('resnet18')

def build_feature_extractor(mode='feature_extraction'):
    base_model = ResNet18(
        input_shape=(224, 224, 3), 
        weights='imagenet', 
        include_top=False
    )
    
    if mode == 'feature_extraction':        # Freeze semua layer ResNet-18
        base_model.trainable = False
    elif mode == 'partial_fine_tuning':     # Unfreeze 5 layer teratas
        base_model.trainable = True
        for layer in base_model.layers[:-5]:
            layer.trainable = False
    elif mode == 'full_fine_tuning':        # Unfreeze semua layer
        base_model.trainable = True

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    return tf.keras.Model(inputs=base_model.input, outputs=x)

# 3. EKSEKUSI TRAINING
results = []
modes = ['feature_extraction', 'partial_fine_tuning', 'full_fine_tuning']

print("\n==========================================")
print("  MULAI PELATIHAN ONE-CLASS RESNET-18")
print("==========================================")

for mode in modes:
    print(f"\n---> Memproses Mode: {mode.upper()}")
    
    extractor = build_feature_extractor(mode)
    features = extract_features(extractor, file_paths)
    
    oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    oc_svm.fit(features)
    
    center = np.mean(features, axis=0)
    distances = np.linalg.norm(features - center, axis=1)
    avg_distance = np.mean(distances)
    
    latency = measure_latency(extractor, file_paths[0])
    plot_and_save_distances(distances, mode)
    
    results.append({
        'Mode Experiment': mode,
        'Jumlah Data Korban': len(file_paths),
        'Avg Feature Distance': f"{avg_distance:.4f}",
        'Latency (ms)': f"{latency:.2f} ms"
    })

df_results = pd.DataFrame(results)
print("\n=== TABEL HASIL ANALISIS DATASET KORBAN (RESNET-18) ===")
print(df_results.to_string(index=False))

df_results.to_csv('results_summary.csv', index=False)
print("\nHasil eksekusi berhasil disimpan ke 'results_summary.csv'")