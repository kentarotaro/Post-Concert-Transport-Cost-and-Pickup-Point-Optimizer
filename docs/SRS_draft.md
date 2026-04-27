# Dokumen Software Requirements Specification (SRS)
**Proyek:** Post-Concert Transport Cost and Pickup Point Optimizer
**Tim:** Request Menu Es Teh Panas
**Tema:** Akomodasi & Mobilitas Penonton Konser di Jakarta

## 1. Daftar Isi
1. Pendahuluan
   1.1 Tujuan
   1.2 Ruang Lingkup (Scope)
2. Deskripsi Umum
   2.1 Perspektif & Fitur Produk
   2.2 Karakteristik Pengguna
   2.3 Lingkungan Operasi
3. Kebutuhan Fungsional
4. Kebutuhan Antarmuka
5. Pembagian Tugas

---

## 2. Pendahuluan

### 2.1 Tujuan
Proyek ini bertujuan untuk membangun sistem berbasis kecerdasan buatan (*AI*) yang mampu menjawab permasalahan mobilitas pasca-konser di Jakarta secara *end-to-end*. Sasaran utamanya meliputi:
* Membangun model *machine learning* (XGBoost) untuk memprediksi *surge multiplier* harga *ride-hailing* berdasarkan variabel kontekstual.
* Mengembangkan sistem estimasi biaya transportasi total untuk berbagai moda dan skenario titik penjemputan.
* Menyediakan rekomendasi titik jemput optimal beserta langkah perjalanan dari *venue* ke titik tersebut.
* Menghasilkan portofolio *AI deployment* yang terdokumentasi lengkap dan dapat direplikasi.

### 2.2 Ruang Lingkup (Scope)
Sistem ini dirancang sebagai alat pendukung keputusan transportasi yang dikhususkan untuk skenario pasca-konser di Jakarta. Batasan sistem mencakup:
* **Fokus Wilayah:** Terbatas pada *venue* besar di Jakarta (GBK, JIS, dan ICE BSD).
* **Data:** Menggunakan data sintetis berbasis logika bisnis harga ojol nyata karena ketiadaan dataset publik yang spesifik.
* **Integrasi:** Tidak mencakup integrasi *real-time* dengan API pihak ketiga (Gojek/Grab), peta interaktif berbasis GPS secara langsung, atau sistem pembayaran nyata.
* **Status Estimasi:** Seluruh angka biaya yang ditampilkan bersifat informatif/simulasi.

---

## 3. Deskripsi Umum

### 3.1 Perspektif & Fitur Produk
Sistem ini mengintegrasikan model prediksi *surge price* dengan algoritma kalkulasi biaya multi-modal. Fungsi utama yang disediakan meliputi:
* **F-01 Surge Price Prediction:** Model XGBoost memprediksi pengali harga (1.0x - 3.5x) untuk ojol dan taksi berdasarkan nama *venue*, jam selesai, kapasitas, cuaca, dan jarak.
* **F-02 Cost Multi-Modal Estimation:** Menghitung estimasi biaya total untuk tiga moda: ojol langsung, ojol setelah jalan kaki ke titik sepi, dan angkutan umum (TransJakarta).
* **F-03 Pickup Point Recommendation:** Rekomendasi 2-3 titik penjemputan alternatif di sekitar *venue* berdasarkan skor jarak jalan kaki dan potensi penurunan harga.
* **F-04 Time-to-Pickup Estimation:** Estimasi waktu total dari keluar *venue* hingga tiba di kendaraan.
* **F-05 Comparative Output Visualization:** Grafik perbandingan horizontal (*bar chart*) biaya vs waktu di antarmuka Gradio.
* **F-06 Natural Language Recommendation:** Penjelasan tertulis mengenai moda terbaik dan alasan pemilihannya dalam bahasa manusia.

### 3.2 Karakteristik Pengguna
Produk ini ditujukan untuk tiga persona utama yang terdampak masalah mobilitas konser:
* **Penonton dari Luar Kota:** Membutuhkan bantuan navigasi dan kepastian biaya karena kurangnya pemahaman tentang medan Jakarta.
* **Penonton Lokal:** Membutuhkan informasi titik jemput resmi agar tidak terjebak kemacetan parah di area utama.
* **Event Organizer (EO):** Membutuhkan alat untuk membantu mengelola arus keluar massa agar tidak terjadi penumpukan di satu titik.

### 3.3 Lingkungan Operasi
Sistem berjalan pada ekosistem berikut:
* **Bahasa & Framework:** Python 3.10+ dengan FastAPI untuk REST API.
* **Model AI:** XGBoost (arsitektur model tersimpan dalam format `.pkl`).
* **Antarmuka:** Gradio UI yang dapat diakses melalui *browser* tanpa instalasi.
* **Deployment:** *Hosting* pada Hugging Face Spaces yang menyediakan URL publik otomatis.

---

## 4. Kebutuhan Fungsional

Deskripsi rinci mengenai perilaku sistem:
* **Input Pengguna:** Nama *venue*, jam selesai (HH:MM), jenis hari (*weekday*/*weekend*), kapasitas penonton, kondisi cuaca, dan zona wilayah tujuan akhir.
* **Logika Pemrosesan:**
    * Sistem melakukan *preprocessing* (normalisasi dan *encoding*) pada input sebelum dimasukkan ke model.
    * Algoritma *cost engine* menghitung biaya menggunakan formula: `base_fare + (jarak_km * tarif_per_km * surge_multiplier)`.
    * *Pickup scorer* melakukan *ranking* terhadap kandidat titik jemput dari dataset statis.
* **Validasi & Keamanan:** Melakukan sanitasi input untuk mencegah *SQL injection* sederhana, *oversized payload*, dan memastikan kebijakan CORS yang aman untuk *deployment* publik.

---

## 5. Kebutuhan Antarmuka

Logika interaksi sistem meliputi:
* **Gradio UI:** Pengguna mengisi formulir *input* secara digital. Output ditampilkan dalam bentuk kartu rekomendasi, tabel rincian biaya/waktu, dan grafik perbandingan visual.
* **REST API:** Menyediakan *endpoint* `POST /predict` yang menerima paket data JSON dan mengembalikan respons terstruktur berisi hasil prediksi dan daftar opsi transportasi.
* **UX Flow:** Pengguna memasukkan data → Sistem memproses (AI + Algoritma) → Sistem merender hasil rekomendasi moda terbaik beserta alasan penghematannya.

---

## 6. Pembagian Tugas

Tim terdiri dari lima mahasiswa Ilmu Komputer yang dibagi berdasarkan spesialisasi divisinya:

| Nama Anggota | Peran | Detail Tugas Utama |
| :--- | :--- | :--- |
| **Kemal** | Ketua Tim & DSAI | Arsitektur *pipeline* AI, *training* model XGBoost, pembuatan antarmuka Gradio, dan koordinasi teknis. |
| **Aliya** | Backend Developer | Setup FastAPI, manajemen *routing* API, integrasi model dengan *server*, serta *deployment* ke Hugging Face. |
| **Gilbard** | UI/UX Designer | Perancangan *wireframe* alur pengguna, panduan visual (warna/tipografi), desain *mockup* output, dan aset presentasi. |
| **Jasmine** | Cyber Security | Validasi input, dokumentasi kebutuhan keamanan di SRS, audit keamanan API, dan validasi kebijakan CORS. |
| **Musqat** | QA & Dokumentasi | Penulisan utama dokumen SRS, penyusunan skenario uji fungsional, pengujian *edge case*, dan penulisan README.md final. |


