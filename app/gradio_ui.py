import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas import PredictRequest
from app.inference import predict_transport


def run_prediction(concert_end_hour, day_type, concert_size, weather,
                   time_since_end_minutes, current_location, destination_zone):
    try:
        request = PredictRequest(
            venue_name="GBK",
            concert_end_hour=int(concert_end_hour),
            day_type=day_type,
            concert_size=concert_size,
            weather=weather,
            time_since_end_minutes=int(time_since_end_minutes),
            destination_zone=destination_zone,
            current_location=current_location,
        )
        result = predict_transport(request)

        surge = result.surge_multiplier
        if surge < 1.5:
            level = "RENDAH"
            color = "green"
        elif surge <= 2.5:
            level = "SEDANG"
            color = "orange"
        else:
            level = "TINGGI"
            color = "red"

        surge_md = f"""
## Surge Multiplier Saat Ini

# {surge}x — Level {level}

> Kondisi: **{weather.upper()}** | Konser **{concert_size.upper()}** | Jam **{concert_end_hour}:00** | {day_type.upper()}
"""

        rows = []
        for opt in result.options:
            is_best = opt.mode == result.best_option
            label = opt.mode.replace("_", " ").title()
            if is_best:
                label = label + " [TERBAIK]"
            rows.append([
                label,
                opt.pickup_point,
                f"{opt.walk_distance_meters} m",
                f"Rp {opt.estimated_cost_idr:,}",
                f"{opt.estimated_time_minutes} menit",
            ])

        headers = ["Moda Transportasi", "Titik Jemput", "Jalan Kaki", "Estimasi Biaya", "Estimasi Waktu"]
        table_data = [headers] + rows

        rec_md = f"""
### Rekomendasi Terbaik

{result.recommendation_text}
"""

        modes  = [o.mode.replace("_", " ").title() for o in result.options]
        costs  = [o.estimated_cost_idr for o in result.options]
        times  = [o.estimated_time_minutes for o in result.options]
        bar_colors = ["#4CAF50" if o.mode == result.best_option else "#2196F3" for o in result.options]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        fig.patch.set_facecolor("#0f0f1a")

        for ax in [ax1, ax2]:
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white", labelsize=9)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")

        bars1 = ax1.barh(modes, costs, color=bar_colors, height=0.5)
        ax1.set_xlabel("Estimasi Biaya (Rp)", color="white")
        ax1.set_title("Perbandingan Biaya", color="white", fontsize=11)
        for bar, val in zip(bars1, costs):
            ax1.text(val * 0.02, bar.get_y() + bar.get_height() / 2,
                     f"Rp {val:,}", va="center", color="white", fontsize=8)

        bars2 = ax2.barh(modes, times, color=bar_colors, height=0.5)
        ax2.set_xlabel("Estimasi Waktu (menit)", color="white")
        ax2.set_title("Perbandingan Waktu", color="white", fontsize=11)
        for bar, val in zip(bars2, times):
            ax2.text(val * 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{val} mnt", va="center", color="white", fontsize=8)

        fig.text(0.5, -0.02, "Hijau = pilihan terbaik  |  Biru = alternatif",
                 ha="center", color="#aaa", fontsize=8)
        plt.tight_layout()

        return surge_md, table_data, rec_md, fig

    except Exception as e:
        return f"Error: {str(e)}", [], "", None


with gr.Blocks(
    title="GBK Transport Optimizer",
    css=".gradio-container { max-width: 1100px; margin: auto; }"
) as demo:

    gr.Markdown("""
    # Post-Concert Transport Cost and Pickup Point Optimizer
    **Gelora Bung Karno, Jakarta** — Prediksi surge price ojol dan rekomendasi rute penjemputan terbaik
    pasca konser menggunakan model XGBoost + algoritma A* (NetworkX).

    ---
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### Input Kondisi Konser")

            concert_end_hour = gr.Slider(
                minimum=19, maximum=24, step=1, value=22,
                label="Jam Selesai Konser",
                info="Semakin larut, surge cenderung lebih tinggi"
            )
            day_type = gr.Dropdown(
                choices=["weekday", "weekend"],
                value="weekend",
                label="Jenis Hari"
            )
            concert_size = gr.Dropdown(
                choices=["small", "medium", "large"],
                value="large",
                label="Ukuran Konser",
                info="small < 10rb | medium 10-40rb | large > 40rb penonton"
            )
            weather = gr.Dropdown(
                choices=["clear", "cloudy", "rain"],
                value="clear",
                label="Kondisi Cuaca"
            )
            time_since_end = gr.Slider(
                minimum=0, maximum=90, step=5, value=10,
                label="Menit Sejak Konser Selesai",
                info="0 = baru selesai, 90 = 1.5 jam setelah konser"
            )
            current_location = gr.Dropdown(
                choices=["Pintu_1_GBK", "Pintu_7_GBK", "Bundaran_Senayan",
                         "MRT_Istora", "FX_Sudirman"],
                value="Pintu_1_GBK",
                label="Posisi Kamu Sekarang di Sekitar GBK"
            )
            destination_zone = gr.Textbox(
                value="Jakarta Selatan",
                label="Zona Tujuan",
                placeholder="Contoh: Jakarta Selatan, Depok, Tangerang"
            )
            btn = gr.Button(
                "Cari Transportasi Terbaik",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=2):
            gr.Markdown("### Hasil Prediksi")
            surge_out = gr.Markdown(value="*Isi form di kiri lalu klik tombol untuk mendapatkan rekomendasi.*")
            table_out = gr.Dataframe(
                headers=["Moda Transportasi", "Titik Jemput", "Jalan Kaki",
                         "Estimasi Biaya", "Estimasi Waktu"],
                label="Perbandingan Opsi Transportasi",
                interactive=False,
                wrap=True
            )
            rec_out   = gr.Markdown()
            chart_out = gr.Plot(label="Grafik Perbandingan Biaya dan Waktu")

    btn.click(
        fn=run_prediction,
        inputs=[concert_end_hour, day_type, concert_size, weather,
                time_since_end, current_location, destination_zone],
        outputs=[surge_out, table_out, rec_out, chart_out]
    )

    gr.Markdown("""
    ---
    *Tim Request Menu Es Teh Panas — OmahTI Internship 2026*
    *Model: XGBoost Regression | Routing: A* Algorithm (NetworkX) | Venue: GBK Jakarta*
    """)


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)