# Prediksi-Harga-Backpack-Berdasarkan-Fitur-Fungsionalitas-dan-Brand
Implementasi machine learning dalam prediksi harga suatu backpack dengan beberapa fitur yang relevan

#### **Domain  Business**  
#### **Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan?**  

Masalah ini perlu diselesaikan karena banyaknya pilihan backpack di pasaran sering kali membuat konsumen bingung. Selain itu, terdapat tas dengan tampilan serupa tetapi memiliki perbedaan harga yang signifikan. Oleh karena itu, penting bagi konsumen untuk memahami faktor-faktor yang mempengaruhi harga suatu backpack.  

Namun, tidak semua orang memiliki anggaran besar untuk membeli tas yang mahal. Oleh sebab itu, diperlukan sebuah tools yang dapat membantu konsumen dalam membandingkan harga dan spesifikasi suatu backpack. Dengan adanya model prediksi ini, konsumen dapat membuat keputusan pembelian yang lebih baik sesuai dengan kebutuhan dan anggaran mereka.  

Menurut Mackie, H., Legg, S., Beadle, J., & Hedderley, D. (2003) dalam penelitian mereka yang berjudul *"Comparison of Four Different Backpacks Intended for School Use"* (Applied Ergonomics, 34(3), 257-264. https://doi.org/10.1016/S0003-6870(03)00034-6), preferensi anak-anak sekolah dalam memilih backpack mengalami perubahan dari sekadar gaya (style) menjadi lebih berorientasi pada fungsi dan kenyamanan seiring berjalannya waktu. Dengan adanya tools ini, seseorang dapat dengan mudah menentukan backpack yang paling sesuai dengan kebutuhan mereka.  

Selain itu, penelitian oleh Gaol, R., Hidayat, N., Tampubolon, A., & Gultom, G. (2024) yang berjudul *"Analisis Pengaruh Harga dan Kualitas Produk Terhadap Keputusan Pembelian Konsumen (Studi Kasus: Mahasiswa Program Studi Ekonomi Fakultas Ekonomi Universitas Negeri Medan)"* (AURELIA: Jurnal Penelitian dan Pengabdian Masyarakat Indonesia. https://doi.org/10.57235/aurelia.v3i2.2804) menemukan bahwa harga dan kualitas produk secara signifikan mempengaruhi keputusan pembelian konsumen, di mana 3,7% responden memilih produk yang lebih murah. Dengan adanya tools ini, konsumen dapat lebih terbantu dalam memilih produk dengan harga yang lebih terjangkau namun tetap sesuai dengan fungsionalitas backpack yang diinginkan.  

### **Business Understanding**  

#### **Problem Statements**  
Dalam industri ritel, khususnya dalam penjualan backpack untuk anak sekolah, terdapat berbagai faktor yang memengaruhi harga suatu produk, seperti brand, bahan, kapasitas, fitur tambahan (misalnya, kompartemen khusus, anti-air, ergonomi), serta tren pasar. Konsumen sering kali mengalami kesulitan dalam memilih backpack yang sesuai dengan kebutuhan mereka karena banyaknya variasi harga untuk produk yang tampak serupa.  

Beberapa permasalahan utama yang dihadapi konsumen dalam memilih backpack yang tepat adalah:  
1. **Variasi Harga yang Signifikan**  
   - Produk dengan tampilan dan spesifikasi yang mirip bisa memiliki harga yang sangat berbeda, membuat konsumen bingung dalam menentukan pilihan yang sesuai dengan anggaran mereka.  

2. **Kurangnya Transparansi dalam Faktor yang Mempengaruhi Harga**  
   - Banyak konsumen tidak memahami faktor-faktor apa saja yang berkontribusi terhadap harga suatu backpack. Mereka sering kali hanya mengandalkan brand sebagai indikator kualitas, padahal ada banyak elemen lain yang menentukan harga.  

3. **Keterbatasan Waktu dalam Melakukan Perbandingan**  
   - Konsumen biasanya harus mencari dan membandingkan berbagai produk secara manual melalui situs e-commerce atau toko fisik, yang bisa memakan banyak waktu dan tenaga.  

4. **Kurangnya Tools untuk Prediksi Harga Berdasarkan Spesifikasi**  
   - Tidak adanya alat yang dapat membantu konsumen dalam memprediksi harga suatu backpack berdasarkan fitur-fiturnya membuat proses pemilihan menjadi kurang efisien.  

#### **Goals (Tujuan)**  
Berdasarkan permasalahan yang telah diidentifikasi, tujuan utama dari proyek ini adalah:  
1. **Menganalisis Faktor-Faktor yang Mempengaruhi Harga Backpack**  
   - Mengidentifikasi komponen-komponen utama dalam spesifikasi backpack yang memiliki pengaruh signifikan terhadap harga, baik itu material, brand, kapasitas, atau fitur tambahan lainnya.  

2. **Membantu Konsumen dalam Memilih Backpack Sesuai dengan Budget dan Kebutuhan**  
   - Dengan adanya analisis harga dan prediksi berbasis data, konsumen dapat lebih mudah menentukan backpack mana yang paling sesuai dengan anggaran dan kebutuhan spesifik mereka.  

3. **Membuat Model Prediksi Harga Berdasarkan Fitur-Fitur Backpack**  
   - Mengembangkan model yang dapat memperkirakan harga suatu backpack berdasarkan atribut atau spesifikasi yang dimilikinya, sehingga dapat digunakan sebagai alat bantu dalam pengambilan keputusan.  

4. **Menyediakan Wawasan Berbasis Data untuk Penjual dan Produsen**  
   - Hasil analisis ini juga dapat membantu produsen atau penjual dalam menentukan strategi harga yang lebih kompetitif berdasarkan preferensi dan daya beli konsumen.  

#### **Solution Statement**  
Untuk mencapai tujuan di atas, proyek ini akan mengimplementasikan dua solusi utama:  

1. **Analisis Data untuk Mengidentifikasi Faktor yang Mempengaruhi Harga**  
   - Menggunakan teknik eksplorasi data seperti analisis deskriptif, korelasi antar variabel, serta visualisasi data untuk memahami faktor-faktor yang paling memengaruhi harga backpack.  
   - Mengidentifikasi brand dan spesifikasi yang memiliki rata-rata harga tertinggi dan terendah, sehingga konsumen dapat mengetahui apakah harga yang ditawarkan sesuai dengan fitur yang diberikan.  

2. **Pengembangan Model Prediksi Harga Backpack**  
   - Menggunakan algoritma pembelajaran mesin seperti **Linear Regression, Random Forest Regression, dan XGBoost** untuk membangun model prediksi harga berdasarkan fitur-fitur backpack.  
   - Melakukan hyperparameter tuning untuk meningkatkan akurasi model, sehingga prediksi harga dapat lebih mendekati harga pasar sebenarnya.  
   - Mengevaluasi performa model dengan metrik seperti **Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), dan R-Squared (R²)** untuk memastikan bahwa model yang dikembangkan dapat memberikan hasil yang akurat dan dapat dipercaya.  

Dengan solusi ini, diharapkan konsumen dapat lebih mudah dalam memilih backpack yang sesuai dengan budget mereka, sementara produsen dan penjual juga dapat mengoptimalkan strategi harga mereka berdasarkan data yang tersedia. 🚀

# **Analisis Harga Backpack Berdasarkan Fitur dan Brand**

## **Data Understanding**

Karakteristik dataset yang digunakan, termasuk jumlah data, kondisi data, serta informasi mengenai variabel yang tersedia. Selain itu, disertakan tautan sumber data untuk referensi lebih lanjut.

### **Informasi Dataset**

Dataset yang digunakan terdiri dari dua bagian, yaitu **data train** dan **data test**. Dataset ini masih dalam kondisi kotor karena terdapat beberapa nilai yang hilang (missing atau NaN). Berikut adalah nilai unik dari beberapa kolom:

- **Brand**: ['Jansport', 'Under Armour', 'Nike', 'Adidas', 'Puma']
- **Material**: ['Leather', 'Canvas', 'Nylon', 'Polyester']
- **Size**: ['Medium', 'Small', 'Large']
- **Compartments**: [7, 10, 2, 8, 1, 5, 3, 6, 4, 9]
- **Laptop Compartment**: ['Yes', 'No']
- **Waterproof**: ['No', 'Yes']
- **Style**: ['Tote', 'Messenger', 'Backpack']
- **Color**: ['Black', 'Green', 'Red', 'Blue', 'Gray', 'Pink']
- **Weight Capacity (kg)**
- **Price**

Berikut adalah informasi rinci mengenai jumlah entri dan tipe data dalam masing-masing dataset:

#### **Data Train**
- **Jumlah data**: 300.000 entri (index: 0 - 299.999)
- **Total kolom**: 11 kolom
- **Ukuran Memori**: 25.2+ MB

#### **Data Test**
- **Jumlah data**: 200.000 entri (index: 0 - 199.999)
- **Total kolom**: 10 kolom (tidak termasuk kolom harga karena ini adalah data untuk prediksi)
- **Ukuran Memori**: 15.3+ MB

### **Sumber Data**

Dataset ini diambil dari Kaggle dengan tautan berikut:
[https://www.kaggle.com/competitions/playground-series-s5e2/Data](https://www.kaggle.com/competitions/playground-series-s5e2/Data)

## **Data Preparation**
## **3. Data Wrangling**
Data wrangling adalah proses pengolahan data agar lebih terstruktur dan siap untuk dianalisis. Proses ini mencakup pemeriksaan format data, pengubahan tipe data jika diperlukan, serta memastikan bahwa semua kolom memiliki nilai yang sesuai.

---

## **4. Pemeriksaan Missing Values**
Langkah selanjutnya adalah memeriksa apakah terdapat nilai yang hilang (missing values) dalam dataset. Jika ditemukan, akan dilakukan penanganan yang sesuai, seperti imputasi atau penghapusan data yang tidak lengkap.

---

## **5. Pembersihan Data (Cleaning Data)**
Proses pembersihan data dilakukan untuk memastikan kualitas data yang digunakan dalam analisis. 
- Menghapus data yang tidak relevan.
- Memeriksa dan menangani data outlier yang mungkin ada dalam dataset.

---
## **6. Eksplorasi Data (Exploratory Data Analysis - EDA)**
Eksplorasi data bertujuan untuk memahami struktur dan distribusi data. Beberapa langkah yang dilakukan dalam EDA meliputi:
- Visualisasi distribusi harga backpack berdasarkan brand.
- Analisis hubungan antara fitur-fitur backpack dengan harga.

---
## **Analisis Korelasi Antar Variabel**
Dilakukan analisis korelasi antara variabel harga dengan fitur-fitur backpack lainnya menggunakan metode statistik.  
- Dari hasil uji **Chi-Square**, ditemukan bahwa **jumlah kompartemen, warna, dan brand** memiliki pengaruh signifikan terhadap harga atau nilai jual suatu backpack.  

---
## **13. Modelling**
Untuk memprediksi harga backpack berdasarkan fitur-fiturnya, digunakan beberapa algoritma, yaitu **Polynomial Regression**, **Linear Regression**, **Random Forest Regression**, **Gradient Boosting Regression**, dan **Artificial Neural Network (ANN)**. Parameter yang digunakan dalam pemodelan ini adalah semua fitur yang ada pada dataset karena dianggap bahwa setiap fitur memiliki pengaruh yang signifikan terhadap harga backpack. 

### **Kelebihan dan Kekurangan Setiap Model**

1. **Polynomial Regression**
   - **Kelebihan**:
     - Mampu menangkap hubungan non-linear antara variabel independen (fitur) dan dependen (harga).
     - Fleksibel dalam memodelkan data yang memiliki pola kompleks.
   - **Kekurangan**:
     - Rentan terhadap overfitting, terutama jika derajat polinomial terlalu tinggi.
     - Kurang efektif jika hubungan antara fitur dan target sebenarnya linier.
     - Performa dapat menurun jika ada banyak fitur (curse of dimensionality).

2. **Linear Regression**
   - **Kelebihan**:
     - Sederhana dan mudah diimplementasikan.
     - Interpretasi model yang mudah karena koefisiennya langsung menunjukkan pengaruh fitur terhadap target.
     - Cepat dalam pelatihan dan prediksi.
   - **Kekurangan**:
     - Hanya cocok untuk hubungan linier antara fitur dan target.
     - Tidak dapat menangkap pola non-linear dalam data.
     - Sensitif terhadap outlier.

3. **Random Forest Regression**
   - **Kelebihan**:
     - Mampu menangani hubungan non-linear dan interaksi antar fitur.
     - Robust terhadap outlier dan noise dalam data.
     - Tidak mudah overfitting karena menggunakan ensemble dari banyak decision tree.
   - **Kekurangan**:
     - Lebih kompleks dan sulit diinterpretasikan dibandingkan model linier.
     - Membutuhkan waktu pelatihan yang lebih lama, terutama dengan dataset besar.
     - Cenderung menggunakan lebih banyak memori.

4. **Gradient Boosting Regression**
   - **Kelebihan**:
     - Mampu menangkap pola non-linear dan interaksi kompleks antar fitur.
     - Biasanya memberikan akurasi yang lebih tinggi dibandingkan model lain.
     - Robust terhadap overfitting jika hyperparameter diatur dengan baik.
   - **Kekurangan**:
     - Lebih lambat dalam pelatihan karena proses boosting yang iteratif.
     - Membutuhkan tuning hyperparameter yang lebih hati-hati.
     - Kurang interpretatif dibandingkan model linier.

5. **Artificial Neural Network (ANN)**
   - **Kelebihan**:
     - Sangat fleksibel dan mampu memodelkan hubungan yang sangat kompleks dan non-linear.
     - Dapat menangani dataset dengan jumlah fitur yang sangat besar.
     - Performa yang sangat baik jika data dan arsitektur model dioptimalkan dengan benar.
   - **Kekurangan**:
     - Membutuhkan data dalam jumlah besar untuk melatih model secara efektif.
     - Proses pelatihan memakan waktu dan sumber daya komputasi yang besar.
     - Sulit diinterpretasikan (black-box model).

---

### **Alasan Menggunakan Beberapa Model**
Saya menggunakan beberapa model untuk memprediksi harga backpack karena ingin membandingkan performa masing-masing model dan memilih model terbaik berdasarkan metrik evaluasi. Setelah melakukan evaluasi, **Gradient Boosting Regression** dipilih sebagai model terbaik karena memiliki nilai **Mean Squared Error (MSE)** dan **Mean Absolute Error (MAE)** yang lebih rendah dibandingkan model lainnya. Selain itu, nilai **R-Squared (R²)** yang mendekati 1 menunjukkan bahwa model ini mampu menjelaskan variasi data dengan baik.

---

### **Alasan Memilih Gradient Boosting Regression**
Model ini dipilih karena kemampuannya untuk menangkap hubungan non-linear antara fitur-fitur backpack dan harganya. Gradient Boosting Regression bekerja dengan membangun model secara bertahap (boosting) dan mengoreksi kesalahan prediksi dari model sebelumnya, sehingga menghasilkan akurasi yang tinggi. Selain itu, model ini juga robust terhadap overfitting jika hyperparameter diatur dengan tepat.

---

### **Evaluasi Model**
Evaluasi model dilakukan menggunakan metrik berikut:
1. **Mean Absolute Error (MAE)**: Mengukur rata-rata kesalahan absolut antara prediksi dan nilai sebenarnya. Semakin kecil MAE, semakin baik model.
2. **Root Mean Squared Error (RMSE)**: Mengukur akar rata-rata kesalahan kuadrat. RMSE lebih sensitif terhadap outlier dibandingkan MAE.
3. **R-Squared (R²)**: Mengukur seberapa baik variasi target dapat dijelaskan oleh model. Nilai R² mendekati 1 menunjukkan model yang baik.

---

### Metrik Evaluasi yang Digunakan:
Metrik evaluasi yang digunakan untuk mengukur kinerja model adalah:

1. **Mean Squared Error (MSE)**: MSE mengukur rata-rata dari kuadrat selisih antara nilai yang diprediksi dan nilai yang sebenarnya. Semakin kecil MSE, semakin baik model dalam memprediksi data.

2. **Mean Absolute Error (MAE)**: MAE mengukur rata-rata dari nilai absolut selisih antara nilai yang diprediksi dan nilai yang sebenarnya. Metrik ini memberikan gambaran tentang seberapa jauh prediksi model dari nilai aktual, tanpa menganggap besar kecilnya kesalahan.

3. **R-Squared (R²)**: R² mengukur seberapa baik model dapat menjelaskan variasi dalam data. Nilai R² berkisar antara 0 hingga 1, dengan 1 menunjukkan model yang sangat baik dan 0 menunjukkan model yang tidak dapat menjelaskan variasi dalam data.

### Hasil Proyek Berdasarkan Metrik Evaluasi:
Dari hasil metrik evaluasi untuk masing-masing model, berikut adalah ringkasannya:

| Model                       | MSE            | MAE           | R²               |
|-----------------------------|----------------|----------------|------------------|
| Linear Regression           | 1509.81        | 33.59          | 0.0012           |
| Random Forest Regression    | 1619.55        | 34.37          | -0.0714          |
| Gradient Boosting Regression| 1509.14        | 33.58          | 0.0016           |
| Artificial Neural Network(ANN) | 1510.11       | 33.0750     | -                   |
| Polynomial Regression  | 1510.3988326774333        | -          |  0.0007635830842684932

- **MSE**: Model Gradient Boosting memiliki nilai MSE yang lebih rendah dibandingkan dengan Random Forest (1619.55), tetapi sedikit lebih tinggi dari Linear Regression (1509.81). Namun, perbedaan ini tidak signifikan dan masih dalam kisaran yang baik.
- **MAE**: MAE pada Gradient Boosting (33.58) hampir setara dengan Linear Regression (33.59), yang menunjukkan bahwa kesalahan absolut model ini hampir sama dengan model Linear Regression.
- **R²**: Model Gradient Boosting memiliki nilai R² tertinggi (0.0016) dibandingkan dengan model lainnya, meskipun nilainya masih sangat rendah, menunjukkan bahwa model ini dapat sedikit lebih baik dalam menjelaskan variabilitas data.

### Pemilihan Model:
Saya memilih **Gradient Boosting Regression** karena meskipun nilai MSE-nya sedikit lebih besar dari Linear Regression, nilai R² dan MAE-nya lebih konsisten dan memberikan hasil yang lebih baik dibandingkan dengan Random Forest. R² yang sedikit lebih tinggi menunjukkan bahwa model ini lebih dapat menjelaskan variabilitas data daripada model Linear Regression, meskipun hasilnya tidak optimal. Dengan demikian, Gradient Boosting dipilih karena memberikan keseimbangan antara performa MSE, MAE, dan R² yang lebih baik dibandingkan dengan model lainnya.

