echo " GBK Concert Transport Optimizer API"
echo " Local Development Server"
echo "========================================"

# pastikan dijalankan dari root project
if [ ! -f "app.py" ]; then
  echo "ERROR: Jalankan script ini dari root direktori project."
  echo "Contoh: bash scripts/run_api.sh"
  exit 1
fi

# cek apakah model sudah tersedia
if [ ! -f "models/surge_predictor.pkl" ]; then
  echo "WARNING: models/surge_predictor.pkl tidak ditemukan."
  echo "Pastikan Kenta sudah menjalankan training pipeline terlebih dahulu."
  exit 1
fi

echo "Model ditemukan. Menjalankan server di http://localhost:7860 ..."
echo "API docs tersedia di http://localhost:7860/docs"
echo "Tekan Ctrl+C untuk menghentikan server."
echo ""

python app.py