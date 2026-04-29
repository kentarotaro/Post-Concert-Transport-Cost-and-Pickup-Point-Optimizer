# Catatan Batasan dan Risiko Sistem
**Penanggung Jawab Draft:** Jasmine & Musqat

## 1. Risiko Keterbatasan Data (Data Limitations)
Model AI memprediksi surge pricing menggunakan **data sintetis** yang di-generate via LLM, bukan data riil historis API Gojek/Grab (karena bersifat privat). Oleh karena itu, surge multiplier yang dihasilkan sistem hanya berupa simulasi logis dan belum tentu akurat 100% di dunia nyata. Estimasi biaya transaksi juga tidak terintegrasi dengan payment gateway asli.

## 2. Keterbatasan Ruang Lingkup (Scope Limitations)
Sebagai produk Minimum Viable Product (MVP), sistem ini bersifat statis dan hanya memetakan 3 venue spesifik di Jakarta (GBK, JIS, ICE BSD). Titik jemput didasarkan pada dataset statis (`venue_pickup_points.json`) dan tidak beradaptasi secara real-time terhadap rekayasa lalu lintas insidental (misal: polisi mendadak menutup Pintu 7 GBK).

## 3. Risiko Kinerja dan Deployment (Operational Risks)
Karena aplikasi di-deploy pada layanan gratis Hugging Face Spaces, terdapat risiko *Cold Start* (server tidur jika lama tidak diakses) yang membuat proses inferensi (prediksi) pertama akan terasa lebih lambat. Selain itu, endpoint bersifat publik tanpa autentikasi, sehingga rentan terhadap spam request jika URL tersebar luas.
