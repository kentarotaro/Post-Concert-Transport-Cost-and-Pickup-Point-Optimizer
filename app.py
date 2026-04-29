import gradio as gr
import uvicorn
from app.main import app as fastapi_app
from app.gradio_ui import demo

# Mounting Gradio inside FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    # Hugging Face menggunakan port 7860 secara default
    uvicorn.run(app, host="0.0.0.0", port=7860)