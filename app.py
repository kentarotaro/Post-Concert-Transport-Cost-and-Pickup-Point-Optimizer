import gradio as gr
from app.main import app as fastapi_app
from app.gradio_ui import demo

# Menempelkan antarmuka Gradio ke root ("/") dari aplikasi FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
