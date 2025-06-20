# Klasifikasi Kualitas Apel dengan KNN Manual

Proyek ini mengimplementasikan algoritma K-Nearest Neighbors (KNN) secara manual untuk mengklasifikasikan kualitas apel berdasarkan berbagai karakteristik fisik. Implementasi dilakukan tanpa menggunakan modul klasifikasi dari scikit-learn.

**Kelompok 1 Kelas A**
1. **Aldo Rifki Firmansyah** - 232410103025  
2. **Adrian Fathan Imama** - 232410103047  
3. **Naufal Rifqi Prasetyo** - 232410103055  
4. **Koniedo Tri Novendar Ramadhan** - 232410103087  
5. **Mohammad Faisal Nur Hidayat** - 232410103091  

## Dataset

https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality?select=apple_quality.csv


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
* Accuracy:  0.9137 (91.38%)
* Precision: 0.9104
* Recall: 0.9173
* F1-Score: 0.9139

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

* K optimal ditemukan = 9
* Akurasi terbaik = 0.9137 (91.38%)
* Performa per kelas:
  * Bad (0): 0.9102 (91.02%)
  * Good (1): 0.9173 (91.73%)

