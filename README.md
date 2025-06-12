# Klasifikasi Kualitas Udara Jakarta dengan KNN Manual

Proyek ini bertujuan untuk mengklasifikasikan kategori kualitas udara di Jakarta menggunakan algoritma K-Nearest Neighbors (KNN) yang **diimplementasikan secara manual** tanpa menggunakan library machine learning seperti `scikit-learn`.

---

### Dataset

Dataset diperoleh dari **[Portal Satu Data DKI Jakarta](https://satudata.jakarta.go.id)** melalui endpoint:

```
https://ws.jakarta.go.id/gateway/DataPortalSatuDataJakarta/1.0/satudata
```

Dataset ini berisi pengukuran harian polutan udara seperti:

* PM10 (`pm_sepuluh`)
* PM2.5 (`pm_duakomalima`)
* Sulfur Dioksida (`sulfur_dioksida`)
* Karbon Monoksida (`karbon_monoksida`)
* Ozon (`ozon`)
* Nitrogen Dioksida (`nitrogen_dioksida`)
* Nilai maksimum (`max`)
* Kategori kualitas udara (`kategori`) seperti: `BAIK`, `SEDANG`, `TIDAK SEHAT`, dll.

---

### Langkah-Langkah Pemrosesan

1. **Pengambilan dan parsing data JSON**
2. **Konversi ke numerik untuk kolom polutan**
3. **Penambahan fitur:**

   * `pm_ratio` = PM2.5 / PM10
   * `gas_ratio` = (SO₂ + NO₂) / (CO + O₃)
4. **Standardisasi manual** (tanpa `StandardScaler`)
5. **Encoding label manual** (tanpa `LabelEncoder`)
6. **Split data manual**: 70% training, 30% testing

---

### KNN Manual

Algoritma KNN diimplementasikan dari nol:

* Menggunakan **jarak Euclidean**
* Menggunakan **bobot inverse distance** agar tetangga lebih dekat memiliki pengaruh lebih besar
* Menentukan **nilai `k` terbaik** dari 1 hingga 13 dengan evaluasi akurasi

---

### Visualisasi

#### 1. Akurasi terhadap nilai `k`:

```python
plt.plot(k_values, accuracies)
```

Menampilkan bagaimana akurasi model berubah sesuai nilai `k`.

#### 2. Confusion Matrix:

Menggambarkan prediksi vs label aktual dalam bentuk tabel heatmap untuk evaluasi model.

---

### Evaluasi

* Akurasi akhir ditampilkan secara manual
* Confusion matrix dihitung dan divisualisasikan
* Semua label dikembalikan ke bentuk kategorinya (`BAIK`, `SEDANG`, dll)

---

### Dependencies

Library yang digunakan:

```bash
pip install pandas matplotlib seaborn requests
```

---

### Contoh Output

```
k Terbaik : 11 accuracy: 0.81

Confusion Matrix:
                SEDANG  BAIK  TIDAK SEHAT  TIDAK ADA DATA
SEDANG             426    17            0               0
BAIK                95   129            0               0
TIDAK SEHAT         11     0            0               0
TIDAK ADA DATA       6     0            0               0

Accuracy: 0.81
```

---

### Catatan

* Dataset harus memiliki distribusi kategori yang mencukupi agar proses pembagian data dan evaluasi berjalan baik.
* Cocok untuk pembelajaran dasar KNN tanpa bergantung pada library ML.

---

### Struktur File

```
project/
│
├── KMEANSUDARAJAKARTA.py  # Script utama
├── README.md              # Dokumentasi ini
├── requirements.txt       # Dokumentasi ini
```
