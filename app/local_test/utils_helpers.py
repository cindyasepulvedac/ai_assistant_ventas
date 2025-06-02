import numpy as np
import streamlit as st 
from streamlit_extras.stylable_container import stylable_container
from streamlit.components.v1 import html
from PIL import Image
from pathlib import Path
import time
import io
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
import base64
from datetime import datetime
import pytz
from dotenv import load_dotenv, find_dotenv

# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image


## Función para cargar imágenes de la galería
def load_gallery_images(gallery_path):
    """Carga las imágenes disponibles en la galería."""
    gallery_path = Path(gallery_path)
    image_files = list(gallery_path.glob("*.jpg")) + list(gallery_path.glob("*.png"))
    return {f.name: f for f in image_files}

def prepare_image(image):
    ## Prepare the image for OpenAI
    ## Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_byte_arr = img_byte_arr.getvalue()

    return img_byte_arr

## Función para procesar mensajes
def process_message(role, content, image=None):
    """Procesa y agrega mensajes al historial."""
    message = {
        "role": role,
        "content": content,
        "timestamp": time.strftime("%H:%M")
    }
    st.session_state.messages.append(message)

# Función para mostrar mensajes
def display_messages():
    """Muestra el historial de mensajes."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "image" in message:
                st.image(message["image"], caption=message["content"])
            st.write(f"{message['content']} - {message['timestamp']}")

