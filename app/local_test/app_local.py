# -------------------------------------------------- #
#              CSS styles                            #
# -------------------------------------------------- #
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
import uuid
from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError, HttpResponseError

from utils_blob_storage import *
from utils_cosmos_db import *
from utils_helpers import *
from utils_ai_prompts import *

## Leer css de estilos
# with open('style.css') as f:
#     st.markdown('<style>{f.read()}</style>', unsafe_allow_html=True)


## Cargar las variables de entorno desde un archivo .env
load_dotenv(find_dotenv(), override=True)

## Definir zona horaria 
colombia_tz = pytz.timezone("America/Bogota")

## Configuración general aplicación
st.set_page_config(
    page_title="Detector de biodiversidad",
    page_icon="🌿",
    layout="centered"
)

# -------------------------------------------------- #
#              CSS STYLES                            #
# -------------------------------------------------- #
## Estilos personalizados
st.markdown("""
    <style>
    .stApp{
        background-color: #eeeeee;
    }
    
    /* Personalización del contenedor principal del chat */
    .main {
        background-color: #eeeeee;
        max-width: 1200px;
        margin: 0 auto;
        font-family: Arial;
        font-size: 12px;
    }
    
    /* Ajustes responsive */
    @media (max-width: 768px) {
        .chat-footer {
            padding: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

title_container_css = """
    
    {
        background-color: #f8f8f8;
        font-family: Arial;
        color: #535353 !important;
        gap: 0px;
        margin-top: -40px;  # Mueve hacia arriba
        padding-bottom: 10px;  # Reduce espacio inferior
        position: relative;
        z-index: 10;
        padding: 15px;
        border-radius: 10px;
        text-align: center !important;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    # div[data-testid="stMarkdownContainer"] {
    #     text-align: center !important;
    #     display: flex;
    #     justify-content: center;
    #     align-items: center;
    # }

     div[data-testid="stHeadingWithActionElements"] {
        text-align: center !important;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    h1 {
        width: 100%;
        text-align: center !important;
        margin: auto;
        font-size: 18px !important; /* Ajusta este valor según necesites */
    }

"""

chat_footer_container_css = """
    div[data-testid="stMarkdownContainer"] .chat-footer-container {
        color: #535353 !important;
    }

"""

message_chat_container_css = """
    /* Estilos para los mensajes */
    div[data-testid="stChatMessage"] {
        background-color: #cfcfcf;
        padding: 10px;
        border-radius: 10px;
        font-family: Arial;
        font-size: 12px;
        letter-spacing: 0px;
    }

"""

chat_input_css = '''
    <style> 
    .stChatInput {
        background-color: #f8f8f8 !important;
        border: 1px solid #ee2b7b !important;
        border-color: #ee2b7b !important
        border-radius: 10px !important;
        color: #535353 !important;
        font-family: Arial;
        font-size: 12px;
    }

    .stChatInput:focus-within,
    .stChatInput [data-baseweb="input"]:focus {
        border-color: #ee2b7b !important;
        outline: none !important;
        box-shadow: 0 0 0 1px #ee2b7b !important;
        border-width: 1px !important;
    }

    .stChatInput textarea:focus {
        border-color: #ee2b7b !important;
    }

    /* Personalización del botón de envío en estado normal */
    .stChatInput button:hover {
        background-color: #ee2b7b !important;
        color: #FFFFFF !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important; /* Transición suave para efectos */
    }
         
    /* Opcional: Cursor de interacción */
    .stButton button {
        cursor: pointer !important;
    }
    </style>
    '''

file_uploader_css = '''
    <style>
     /* Estilos para el file uploader */
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
        color: #ee2b7b !important;
        border-radius: 8px !important;
        font-family: Arial;
        font-size: 14px;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
        padding: 0 !important;
        font-family: Arial;
        font-size: 14px;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: #ee2b7b !important;
        # border: 1px solid #ee2b7b !important;
        border-color: #ee2b7b !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploader"] button {
        # border: 1px solid #ee2b7b !important;
        border-color: #ee2b7b !important;
        color: #ee2b7b !important;
    }

    [data-testid="stFileUploader"] button:active,
    [data-testid="stFileUploader"] button:focus {
        background-color: #ee2b7b !important;
        color: white !important;
        border-color: #ee2b7b !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: rgba(238, 43, 123, 0.1) !important;
        border-color: #ee2b7b !important;
    }
    
    /* Ocultar elementos innecesarios del uploader */
    [data-testid="stFileUploader"] section > input + div,
    [data-testid="stFileUploader"] section + div {
        display: none !important;

    }
        
    </style>
    '''

# -------------------------------------------------- #
#              ENV VARIABLES                         #
# -------------------------------------------------- #
AZURE_OPENAI_ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_LLM_DEPLOYMENT=os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT")
AZURE_OPENAI_API_VERSION=os.getenv("AZURE_OPENAI_API_VERSION")

AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_CLIENT_ID_KEYVAULT = os.getenv("AZURE_CLIENT_ID_KEYVAULT")
AZURE_CLIENT_SECRET_KEYVAULT = os.getenv("AZURE_CLIENT_SECRET_KEYVAULT")
AZURE_TENANT_ID_KEYVAULT = os.getenv("AZURE_TENANT_ID_KEYVAULT")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
VAULT_URI = os.getenv("VAULT_URI")

AZURE_CLIENT_ID_SP = os.getenv("AZURE_CLIENT_ID_SP")
AZURE_TENANT_ID_SP = os.getenv("AZURE_TENANT_ID_SP")
AZURE_CLIENT_SECRET_SP = os.getenv("AZURE_CLIENT_SECRET_SP")
AZURE_SP_NAME = os.getenv("AZURE_SP_NAME")

AZURE_COSMOSDB_DATABASE_NAME = os.getenv("AZURE_COSMOSDB_DATABASE_NAME")
AZURE_COSMOSDB_CONTAINER_NAME = os.getenv("AZURE_COSMOSDB_CONTAINER_NAME")
AZURE_COSMOSDB_URI = os.getenv("AZURE_COSMOSDB_URI")
AZURE_COSMOSDB_DATABASE_ACCOUNT = os.getenv("AZURE_COSMOSDB_DATABASE_ACCOUNT")

AZURE_COSMOSDB_URI_SECRET_ID = os.getenv("AZURE_COSMOSDB_URI_SECRET_ID")
AZURE_COSMOSDB_URI_SECRET_NAME = os.getenv("AZURE_COSMOSDB_URI_SECRET_NAME")
AZURE_COSMOSDB_KEY_SECRET_ID = os.getenv("AZURE_COSMOSDB_KEY_SECRET_ID")
AZURE_COSMOSDB_KEY_SECRET_NAME = os.getenv("AZURE_COSMOSDB_KEY_SECRET_NAME")

AZURE_BLOB_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_BLOB_STORAGE_ACCOUNT_NAME")
AZURE_BLOB_STORAGE_CONTAINER_NAME = os.getenv("AZURE_BLOB_STORAGE_CONTAINER_NAME")
AZURE_BLOB_STORAGE_CONNECTION_STRING = os.getenv("AZURE_BLOB_STORAGE_CONNECTION_STRING")
AZURE_BLOB_STORAGE_ACCOUNT_URL = os.getenv("AZURE_BLOB_STORAGE_ACCOUNT_URL")

# -------------------------------------------------- #
#           AZURE SERVICES CONNECTION                #
# -------------------------------------------------- #

cosmos_client = get_cosmos_client(AZURE_CLIENT_ID_KEYVAULT, AZURE_CLIENT_SECRET_KEYVAULT, AZURE_TENANT_ID_KEYVAULT, VAULT_URI, AZURE_COSMOSDB_KEY_SECRET_NAME, AZURE_COSMOSDB_URI_SECRET_NAME)
container_client = get_container_client(AZURE_CLIENT_ID_SP, AZURE_CLIENT_SECRET_SP, AZURE_TENANT_ID_SP, AZURE_BLOB_STORAGE_ACCOUNT_URL, AZURE_BLOB_STORAGE_CONTAINER_NAME)

# -------------------------------------------------- #
#              BACK LOGIC                            #
# -------------------------------------------------- #

## Inicialización del modelo de lenguaje Azure OpenAI
model = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    deployment_name=AZURE_OPENAI_LLM_DEPLOYMENT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.5
)

## Inicialización del estado de la sesión
if "messages" not in st.session_state:
    st.session_state.messages = [
                                {"role": "assistant", 
                                  "content": "¡Hola! Soy tu Detector de Biodiversidad de Comfama :robot_face:. Compárteme tus fotografías de especies avistadas en nuestros Parques Bosques y te ayudaré a identificarlas. También puedo brindar respuestas a tus preguntas sobre biodiversidad.",
                                  "conversation_id": str(datetime.now(colombia_tz).strftime("%Y-%m-%dT%H:%M:%SZ"))}
                                ]

# -------------------------------------------------- #
#              FRONT LOGIC                           #
# -------------------------------------------------- #

## Layout principal
def main():

    ## Appbar logo
    st.logo(r'app/assets/images/logo_comfama.png', size="small", icon_image=r'app/assets/images/logo_comfama.png') ##Así se requieren las rutas para la webapp
    # st.logo(r'app\assets\images\logo_comfama.png', size="small", icon_image=r'app\assets\images\logo_comfama.png') ##Así se requieren las rutas para las pruebas locales
    
    with stylable_container(key='title_container', css_styles=title_container_css):

        # st.markdown(f'<h1 style="text-align:center;color:#535353;font-size:20px;font-family:Arial;">{"Detector de biodiversidad🌿"}</h1>', unsafe_allow_html=True)
        st.title("Detector de biodiversidad")
        
    ## Contenedor principal del chat
    with stylable_container(key='chat_footer_container', css_styles=chat_footer_container_css):
    
        with st._bottom:
            ## Crear columnas para input y uploader
            col1, col2 = st.columns([0.8, 0.2])

            ## Input de texto
            with col1:
                user_input = st.chat_input("Escribe tu mensaje aquí...")
                st.markdown(chat_input_css, unsafe_allow_html=True)

            ## Uploader de imágenes
            with col2:
                uploaded_file = st.file_uploader(
                    "Cargar imagen",
                    type=['png', 'jpg', 'jpeg'],
                    label_visibility="collapsed",
                    key="image_upload",
                    help="Cargar imagen"
                )
                st.markdown(file_uploader_css, unsafe_allow_html=True)

    
        with stylable_container(key='inner_chat_container', css_styles=message_chat_container_css):

            ## Mostrar mensajes anteriores en el chat
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            ## Procesar entrada de usuario
            if user_input:

                my_uuid = uuid.uuid4()
                fecha = str(datetime.now(colombia_tz).strftime("%Y-%m-%dT%H:%M:%SZ"))

                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.write(user_input)


                ## Lógica para procesar el mensaje y generar respuesta
                try: 
                    with st.spinner("Cargando respuesta..."):
                        messages = generate_ai_response(user_input)
                        response = model.invoke(messages)

                    ## Display results
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                    with st.chat_message("assistant"):
                        st.write(response.content)

                    ## Almacenamiento de interacciones
                    new_item = {
                        "id": str(my_uuid),
                        "interaction_date": str(fecha),
                        "interacion_type": 'Pregunta',
                        "user_input":str(user_input),
                        "ai_response": str(response.content),
                        "token_usage": response.response_metadata['token_usage'],
                        "security_filters": response.response_metadata['prompt_filter_results'],
                    }
                    try:
                        insert_dict_to_cosmosdb(cosmos_client, AZURE_COSMOSDB_DATABASE_NAME, AZURE_COSMOSDB_CONTAINER_NAME, new_item)
                    except ValueError as e:
                        logging.error(f"Error de validación: {str(e)}")

                    except ResourceNotFoundError as e:
                        logging.error(f"Base de datos o contenedor no encontrado: {str(e)}")

                    except HttpResponseError as e:
                        logging.error(f"Error de comunicación con Cosmos DB: {str(e)}")

                    except Exception as e:
                        logging.error(f"Error inesperado al insertar documento: {str(e)}")
                
                except:
                    response = 'No puedo encontrar una respuesta. Por favor, replantea la pregunta e intenta nuevamente.'
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.write(response)


            ## Procesar imagen subida
            elif uploaded_file is not None:

                image = Image.open(uploaded_file)
                my_uuid = uuid.uuid4()
                fecha = str(datetime.now(colombia_tz).strftime("%Y-%m-%dT%H:%M:%SZ"))
                blob_name = str(my_uuid) + '.png'
                
                st.session_state.messages.append({"role": "user", "content": image})
                with st.chat_message("user"):
                    st.write(image)

                ## Lógica para procesar la imagen y generar respuesta
                try:
                    img_byte_arr = prepare_image(image)

                    with st.spinner("Analizando imagen..."):
                        classify_message = classify_image_ai(image, img_byte_arr)
                        classify_response = model.invoke(classify_message)
                        print(classify_response.content.lower())

                        if classify_response.content.lower() in ['animal', 'planta', 'hongo']:
                            messages = identify_image_ai(image, img_byte_arr)
                            response = model.invoke(messages)

                    ## Display results
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                    with st.chat_message("assistant"):
                        st.write(response.content)
                        
                    ## Almacenamiento de imagen
                    try:
                        upload_img_to_adls(container_client, blob_name, image)
                    except Exception as e:
                        logging.error(f"Error al almacenar la imagen en ADLS: {str(e)}")

                    ## Almacenamiento de respuesta
                    new_item = {
                            "id": str(my_uuid),
                            "interaction_date": str(fecha),
                            "interacion_type": 'Identificación imagen',
                            "user_input":'Imagen',
                            "ai_response": str(response.content),
                            "token_usage": response.response_metadata['token_usage'],
                            "security_filters": response.response_metadata['prompt_filter_results'],
                        }
                    try:
                        insert_dict_to_cosmosdb(cosmos_client, AZURE_COSMOSDB_DATABASE_NAME, AZURE_COSMOSDB_CONTAINER_NAME, new_item)
                    except ValueError as e:
                        logging.error(f"Error de validación: {str(e)}")

                    except ResourceNotFoundError as e:
                        logging.error(f"Base de datos o contenedor no encontrado: {str(e)}")

                    except HttpResponseError as e:
                        logging.error(f"Error de comunicación con CosmosDB: {str(e)}")

                    except Exception as e:
                        logging.error(f"Error inesperado al insertar el registro en CosmosDB: {str(e)}")

                except:
                    response = 'Lo siento, no puedo identificar la imagen. Sólo puedo identificar imágenes relacionadas con biodiversidad y naturaleza. Por favor, intenta nuevamente con una otra imagen.'
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.write(response)
            
            else:
                pass

    # Punto de entrada principal
if __name__ == "__main__":
    main()