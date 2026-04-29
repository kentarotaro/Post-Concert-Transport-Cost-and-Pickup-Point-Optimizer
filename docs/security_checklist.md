
# API Security Checklist

**Penanggung Jawab:** Jasmine 

**Target File:** `app/main.py`, `app/schemas.py`



## 1. Input Validation & Type Checking

- [ ] `venue_name`: Wajib divalidasi hanya menerima nilai "GBK", "JIS", atau "ICE BSD" (Cegah input sembarangan).

- [ ] `concert_end_hour`: Validasi rentang angka hanya antara 19 hingga 24 (Tipe data wajib integer).

- [ ] `day_type`: Hanya menerima "weekday" atau "weekend".

- [ ] `concert_size`: Hanya menerima "small", "medium", atau "large".

- [ ] `weather`: Hanya menerima "clear", "cloudy", atau "rain".

- [ ] `destination_zone`: Dibatasi pada daftar zona yang valid di schema.

- [ ] Batas Karakter: Pastikan tidak ada input string yang melebihi batas wajar (misal > 50 karakter) untuk mencegah oversized payload.



## 2. Injection & Sanitization

- [ ] Walaupun menggunakan XGBoost (bukan database SQL), pastikan input di-sanitize dari karakter khusus untuk mencegah potensi XSS jika output di-render mentah ke antarmuka web.



## 3. Server & CORS Configuration

- [ ] CORS Policy: Pastikan `allow_origins` pada FastAPI dikonfigurasi dengan aman. Boleh `["*"]` untuk keperluan public API di Hugging Face, tapi berikan catatan risikonya.

- [ ] Error Handling: Pastikan pesan error internal (500 Internal Server Error) tidak membocorkan stack trace atau struktur direktori server ke pengguna luar. Tampilkan pesan error umum yang rapi.

