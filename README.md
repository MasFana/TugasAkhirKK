# Klasifikasi Kualitas Apel dengan KNN Manual

Proyek ini mengimplementasikan algoritma K-Nearest Neighbors (KNN) secara manual untuk mengklasifikasikan kualitas apel berdasarkan berbagai karakteristik fisik. Implementasi dilakukan tanpa menggunakan modul klasifikasi dari scikit-learn.

## Dataset

Dataset berisi pengukuran karakteristik apel dengan fitur-fitur:

* Size (Ukuran)
* Weight (Berat)
* Sweetness (Tingkat Kemanisan)
* Crunchiness (Kerenyahan)
* Juiciness (Kejusian)
* Ripeness (Kematangan)
* Acidity (Keasaman)
* Quality (Label: Good/Bad)

## Langkah-Langkah Pemrosesan

1. **Preprocessing Data**
   * Penghapusan kolom tidak penting (A_id)
   * Konversi label ke numerik (Good=1, Bad=0)
   * Penanganan missing values dengan median
   * Normalisasi fitur dengan MinMax Scaling (0-1)

2. **Analisis Statistik Deskriptif**
   * Perhitungan statistik dasar (mean, median, std, dll)
   * Analisis outlier
   * Visualisasi distribusi data
   * Analisis korelasi antar fitur

3. **Implementasi KNN**
   * Perhitungan jarak Euclidean
   * Pembobotan inverse distance
   * Pencarian k tetangga terdekat
   * Prediksi kelas berdasarkan majority voting

## Visualisasi

1. **Before-After Scaling**
   * Boxplot data sebelum dan sesudah normalisasi
   * Deteksi outlier visual

2. **Analisis Korelasi**
   * Heatmap korelasi antar fitur
   * Scatter plot fitur vs target

3. **Evaluasi Model**
   * Plot akurasi vs nilai k
   * Confusion matrix
   * Distribusi prediksi

## Evaluasi

Metrik evaluasi yang digunakan:
* Accuracy: {final_accuracy:.2f}
* Precision: {precision:.2f}
* Recall: {recall:.2f}
* F1-Score: {f1_score:.2f}

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Struktur File

```
project/
│
├── KKApel.ipynb          # Notebook utama
├── apple_quality.csv     # Dataset
├── README.md            # Dokumentasi
└── requirements.txt     # Dependency list
```

## Hasil Analisis

* K optimal ditemukan = {best_k}
* Akurasi terbaik = {final_accuracy:.2f}
* Performa per kelas:
  * Bad (0): {bad_accuracy:.2f}%
  * Good (1): {good_accuracy:.2f}%
