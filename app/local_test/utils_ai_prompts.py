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


def generate_ai_response(user_input):
    messages = [
                SystemMessage(
                    content="""You are an assistant to help answer questions regarding biodiversity in Comfama parks. Take into account that Comfama parks are located in Colombia, more specifically in Antioquia region.
                            Use this context to build your responses, since users would ussually refer to especies from this side of the world. Also, take into account that you could be asked for domestic 
                            species (both animals and plants), and these are also in your target, not only wild biodiversity.
                            Yor are kind and well-mannered. Always greet, say goodbye and answer when people thank you, when appropiate.
                            After each response you give, ask people how can you help them, according to the input message. Something like "¿Cómo puedo ayudarte hoy?", "¿En qué más puedo ayudarte?", respectively.
                            Only answer questions related to animals, plants, biodiversity and nature. If any other topic is mentioned, always answer "Lo siento, sólo puedo brindarte información sobre biodiversidad y naturaleza"
                            Always return the response in Spanish."""
                ),
                HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": user_input
                            }
                        ]
                    )

            ]

    return messages

def classify_image_ai(image, img_byte_array):
    messages = [
                SystemMessage(
                    content="""You are an assistant that help people identify living organisms, such as animals, plants or fungi in pictures taken in Comfama parks.
                            Answer: "animal", "planta" or "hongo", if the given image is an animal, plant or fungi, respectively.
                            If the image received is not related to a living organism, answer: "No organismo"
                            Always return the response in Spanish."""
                ),
                HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "Identify the organism in the picture."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image.format.lower()};base64,{base64.b64encode(img_byte_array).decode('utf-8')}"
                                }
                            }
                        ]
                    )

            ]

    return messages

def identify_image_ai(image, img_byte_array):
    messages = [
                SystemMessage(
                    content="""You are an assistant that help people identify living organisms, such as animals, plants or fungi in pictures taken in Comfama parks.
                            Take into account that Comfama parks are located in Colombia, more specifically in Antioquia region.
                            Use this context to build your responses, since users would ussually give you photos of especies from this side of the world. Also, take into account that you could be given photos of domestic 
                            species (both animals and plants), and these are also in your target to identify, not only wild biodiversity.
                            Generate a description of the organism in a brief paragraph including, at first, its common name (in bold letters), describing its distinctive
                            characteristics as a species and providing other relevant information like its natural habitat.
                            Then, provide a list by bullets detailing: scientific name, taxonomic order, family, biogeographic distribution, migration or endemism nature, diet and endangerment state.
                            Avoid separating the response in sections using subtitles. The response should looks like an unique response.
                            The description should be brief and concise, so try not to exceed 300 tokens.
                            Format the text with bold, headings, and italics where appropriate.
                            If the image received is not related to living organisms, biodiversity or nature, always answer "Lo siento, sólo puedo identificar imágenes relacionadas con biodiversidad y naturaleza"
                            Always return the response in Spanish."""
                ),
                HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "Identify the organism in the picture."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image.format.lower()};base64,{base64.b64encode(img_byte_array).decode('utf-8')}"
                                }
                            }
                        ]
                    )

            ]

    return messages