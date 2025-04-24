## 📘 Tahapan Pembuatan Model Klasifikasi (Decision Tree)

### 🔍 Deskripsi Masalah
Tujuan dari proyek ini adalah membuat model klasifikasi untuk menentukan apakah seseorang **layak mendapatkan kredit komputer** atau tidak, berdasarkan atribut-atribut seperti umur, penghasilan, status mahasiswa, dan rating kredit. Model yang digunakan adalah **Decision Tree**.

---

### ✅ Langkah-Langkah

#### 1. Load Dataset
Dataset diunduh dari Google Drive dan dimuat menggunakan `pandas`. Dataset berisi data dummy kredit komputer dengan kolom-kolom kategorikal dan target `Buys_Computer`.

#### 2. Preprocessing Data
Karena banyak fitur berupa kategori (seperti `Age`, `Income`, dll), maka semua fitur kategorikal dikonversi menjadi numerik menggunakan **Label Encoding** agar bisa digunakan oleh model klasifikasi.

#### 3. Pemisahan Fitur dan Target
Fitur (`X`) diambil dari kolom:
- `Age`
- `Income`
- `Student`
- `Credit_Rating`

Target (`y`) adalah kolom:
- `Buys_Computer` (0 = tidak layak, 1 = layak)

#### 4. Split Data Training dan Testing
Dataset dibagi menjadi dua bagian:
- **80% untuk training**
- **20% untuk testing**
Menggunakan `train_test_split` dari `sklearn`.

#### 5. Pembuatan dan Pelatihan Model
Model **Decision Tree** dibuat dengan `DecisionTreeClassifier` dari scikit-learn dan dilatih menggunakan data training.

#### 6. Prediksi dan Evaluasi Model
Model digunakan untuk memprediksi data test. Evaluasi dilakukan menggunakan:
- **Akurasi**
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1-Score)**

#### 7. Visualisasi Decision Tree
Pohon keputusan divisualisasikan menggunakan `plot_tree` untuk memahami bagaimana model membuat keputusan berdasarkan fitur-fitur yang ada.

---

### 📊 Hasil Evaluasi Model Decision Tree

Model Decision Tree digunakan untuk melakukan klasifikasi apakah seseorang **layak atau tidak layak mendapatkan kredit komputer**, berdasarkan beberapa fitur seperti usia, status pelajar, pendapatan, dan rating kredit.

### 🔍 Informasi Dataset
- **Total Data**: 1001 baris
- **Data Latih (Training)**: 80% (800 data)
- **Data Uji (Testing)**: 20% (201 data)

### 🌟 Akurasi Model
Model menghasilkan akurasi sebesar **80.50%**, yang berarti model mampu mengklasifikasikan data uji dengan tingkat ketepatan sebesar itu.

### 📊 Confusion Matrix
Confusion Matrix menunjukkan jumlah prediksi benar dan salah dalam bentuk matriks:

|                      | Prediksi: Tidak | Prediksi: Ya |
|----------------------|-----------------|---------------|
| **Asli: Tidak**      |      TP         |      FN       |
| **Asli: Ya**         |      FP         |      TN       |

Contoh interpretasi:
- **TP (True Positive)**: Orang yang memang tidak layak dan diprediksi tidak layak.
- **TN (True Negative)**: Orang yang layak dan diprediksi layak.
- **FP/FN (False)**: Kesalahan prediksi.

### 📁 Classification Report
Classification report memberikan nilai:
- **Precision**: Ketepatan prediksi kelas positif
- **Recall**: Kemampuan mendeteksi semua data kelas positif
- **F1-score**: Rata-rata harmonis dari precision dan recall

Contoh (ubah dengan hasil asli):
```
=== Evaluasi Model Decision Tree ===
Akurasi: 80.50%

Confusion Matrix:
 [[ 57  14]
 [ 25 104]]

Classification Report:
               precision    recall  f1-score   support

       Tidak       0.70      0.80      0.75        71
          Ya       0.88      0.81      0.84       129

    accuracy                           0.81       200
   macro avg       0.79      0.80      0.79       200
weighted avg       0.82      0.81      0.81       200
```

---

### 🌳 Visualisasi Pohon Keputusan
Visualisasi pohon keputusan memberikan gambaran bagaimana data diproses dan diputuskan oleh model. Setiap cabang mewakili kondisi logika (contoh: `Income <= 1.5`), dan setiap daun (akhir cabang) menyatakan keputusan akhir model (Ya/Tidak).

Pohon ini membantu memahami **alur logika keputusan** model secara transparan dan dapat dijelaskan ke stakeholder non-teknis.

---

### 📦 Tools dan Library
- Python
- pandas
- scikit-learn
- matplotlib

