```
# Dokumen Skenario Pengujian (Test Cases)

Dokumen ini berisi skenario pengujian untuk memastikan API berjalan dengan baik dan mengembalikan respons yang sesuai.
```
---

## Test Case 01: Memeriksa Status Kesehatan API
**Endpoint:** `/health`
**Tujuan:** Memastikan server dan model *machine learning* sudah dimuat dan berjalan (*running*).

**Request:**
```bash
curl --location '/health'
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## Test Case 02: Meminta Rekomendasi Rute dan Prediksi Harga
**Endpoint:** `/predict`
**Tujuan:** Menguji apakah model XGBoost dan algoritma pencarian rute dapat memproses input variabel konser secara normal dan menghasilkan 3 opsi perjalanan.

**Request:**
```bash
curl --location '/predict' \
--header 'Content-Type: application/json' \
--data '{
  "venue_name": "GBK",
  "concert_end_hour": 19,
  "day_type": "weekday",
  "concert_size": "small",
  "weather": "clear",
  "time_since_end_minutes": 0,
  "destination_zone": "Jakarta Selatan",
  "current_location": "Pintu_1_GBK"
}'
```

**Expected Response:**
```json
{
  "surge_multiplier": 1.11,
  "best_option": "transjakarta",
  "recommendation_text": "Surge saat ini 1.11x. Jarak ke Jakarta Selatan: 8.0 km. Rekomendasi terbaik: Transjakarta — jalan kaki 530m ke Stasiun MRT Istora, estimasi biaya Rp 3,500 dengan waktu tempuh sekitar 23 menit. Hemat Rp 27,700 dibanding naik ojol langsung dari venue.",
  "options": [
    {
      "mode": "ojol_langsung",
      "pickup_point": "Pintu 1 GBK",
      "walk_distance_meters": 0,
      "estimated_cost_idr": 31200,
      "estimated_time_minutes": 19
    },
    {
      "mode": "ojol_jalan_dulu",
      "pickup_point": "Pintu 7 GBK",
      "walk_distance_meters": 260,
      "estimated_cost_idr": 29000,
      "estimated_time_minutes": 22
    },
    {
      "mode": "transjakarta",
      "pickup_point": "Stasiun MRT Istora",
      "walk_distance_meters": 530,
      "estimated_cost_idr": 3500,
      "estimated_time_minutes": 23
    }
  ]
}
