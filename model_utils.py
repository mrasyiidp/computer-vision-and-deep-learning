import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Memuat dan memproses gambar menggunakan PIL & NumPy tanpa tensorflow.keras"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0  # Preprocessing MobileNetV2
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def extract_features(model, file_paths):
    """Mengekstrak fitur vektor dari daftar gambar"""
    features = []
    for path in file_paths:
        img_array = load_and_preprocess_image(path)
        feat = model.predict(img_array, verbose=0)
        features.append(feat.flatten())
    return np.array(features)

def measure_latency(feature_extractor, sample_img_path, num_samples=30):
    """Mengukur latensi rata-rata ekstrasi fitur (ms/frame)"""
    img_array = load_and_preprocess_image(sample_img_path)
    _ = feature_extractor.predict(img_array, verbose=0)  # Warm-up
    
    start_time = time.time()
    for _ in range(num_samples):
        _ = feature_extractor.predict(img_array, verbose=0)
    end_time = time.time()
    
    avg_latency_ms = ((end_time - start_time) / num_samples) * 1000
    return avg_latency_ms

def plot_and_save_distances(distances, mode_name):
    """Menyimpan grafik sebaran skor jarak fitur"""
    plt.figure(figsize=(8, 5))
    plt.plot(distances, marker='o', linestyle='-', color='b', label='Jarak Fitur ke Pusat Cluster')
    plt.title(f'Feature Distance Score - Mode: {mode_name}')
    plt.xlabel('Indeks Sampel')
    plt.ylabel('Euclidean Distance')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'plot_{mode_name}.png')
    plt.close()
    print(f"Grafik berhasil disimpan sebagai 'plot_{mode_name}.png'")