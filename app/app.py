# -------------------------------------------------- #
#           IMPORT MODULES                           #
# -------------------------------------------------- #
## -- AI services libraries
import streamlit as st 
from streamlit_extras.stylable_container import stylable_container
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS

## -- Basic dependencies
from PIL import Image
import os
from datetime import datetime
import pytz
from dotenv import load_dotenv, find_dotenv
import uuid
import logging
import toml

## -- Custom modules
from utils_ai_prompts import *
from utils_helpers import *

# --- Configurar Logging Básico ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------------------------- #
#            APP INITIALIZATION                      #
# -------------------------------------------------- #
## --- Configuración general aplicación
st.set_page_config(
    page_title="🤖 Asistente Venta Empresarial",
    layout="centered"
)

## --- Definir zona horaria 
colombia_tz = pytz.timezone("America/Bogota")

# --- Inicialización del estado de la sesión (unificado) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": """
                    👋 **¡Hola! Soy tu Asistente para la Venta Empresarial de Comfama.**

                    Estoy aquí para apoyarte con información basada en los datos de ventas tabulares de empresas compradoras 🏢📊.

                    Puedes preguntarme cosas como:

                    - 🤖 *¿Cuáles son los arquetipos más comunes de empresas compradoras?*
                    - 📦 *¿Qué tipo de productos puedo ofrecer a las empresas promotoras del movimiento?*
                    - 💰 *Dame el total de ventas disponible para la empresa 11*
                    - 🩺 *¿Qué le puedo ofrecer a empresas interesadas en programas de salud y cuidado de sus empleados?*

                    ¡Estoy listo para ayudarte! 😊
                """,
         "timestamp": datetime.now(colombia_tz).isoformat()} # Usar ISO para consistencia
    ]


# -------------------------------------------------- #
#              BACK LOGIC                            #
# -------------------------------------------------- #
load_dotenv(find_dotenv(), override=True)

@st.cache_resource
def load_ai_resources():
    try:
        # TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
        # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        config = toml.load("config.toml")
        TOGETHER_API_KEY = config["api"]["together_api_key"]
        GOOGLE_API_KEY = config["api"]["google_api_key"]
        FAISS_INDEX_PATH = r"app\data\faiss_index_tabular"

        if not TOGETHER_API_KEY or not GOOGLE_API_KEY:
            st.error("Faltan las API Keys necesarias. Verifica TOGETHER_API_KEY y GOOGLE_API_KEY.")
            st.stop()

        qa_chain = load_qa_chain(
            faiss_path=FAISS_INDEX_PATH,
            together_api_key=TOGETHER_API_KEY,
            google_api_key=GOOGLE_API_KEY
        )

        if qa_chain is None:
            st.error("No se pudo inicializar el asistente de IA.")
            return None

        return qa_chain

    except Exception as e:
        st.error(f"Error al cargar componentes de IA: {e}")
        return None

# --- Cargar recursos AI y crear cadena QA ---
qa_chain_instance = load_ai_resources()

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

    div[data-testid="stChatMessage"] {
        background-color: #cfcfcf;
        padding: 10px;
        border-radius: 10px;
        font-family: Arial;
        font-size: 12px;
        letter-spacing: 0px;
    }

    .chat-centered-img {
        display: flex;
        justify-content: center;
    }

    .chat-centered-img img {
        border-radius: 8px;
        margin-top: 6px;
    }

"""

chat_input_css = """
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
    """

file_uploader_css = """
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
    """



# -------------------------------------------------- #
#              FRONT LOGIC                           #
# -------------------------------------------------- #

## Layout principal
def main():

    ## Appbar logo
    st.logo(r'app/assets/images/logo_comfama.png', size="small", icon_image=r'app/assets/images/logo_comfama.png') 
    with stylable_container(key='title_container', css_styles=title_container_css):
        st.title("🤖 Asesor AI Venta Empresarial")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) 
        
    # --- Input del usuario y procesamiento ---
    if user_prompt := st.chat_input("Escribe tu pregunta aquí...", disabled=not qa_chain_instance): # Deshabilitar si la cadena no cargó
        # Añadir mensaje del usuario al historial y mostrarlo
        st.session_state.messages.append({
            "role": "user",
            "content": user_prompt,
            "timestamp": datetime.now(colombia_tz).isoformat()
        })

        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Procesar con la cadena QA si está disponible
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    response = qa_chain_instance.invoke({"query": user_prompt})
                    answer = response["result"]
                    source_documents = response.get("source_documents", [])

                    st.markdown(answer) # Mostrar respuesta del asistente

                    # Añadir respuesta del asistente al historial
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "timestamp": datetime.now(colombia_tz).isoformat(),
                        "sources": source_documents # Opcional: guardar fuentes con el mensaje
                    })
                    
                    # Opcional: Mostrar fuentes directamente bajo la respuesta si lo prefieres
                    if source_documents:
                        with st.expander("Ver fuentes de información"):
                            for i, doc in enumerate(source_documents):
                                source_name = doc.metadata.get('source_filename', doc.metadata.get('source', 'desconocido'))
                                row_idx = doc.metadata.get('row_index', 'N/A')
                                st.caption(f"Fuente {i+1} (Archivo: '{source_name}', Fila Aprox: {row_idx})")
                                st.markdown(f"> {doc.page_content[:250]}...") # Mostrar un extracto
                                st.markdown("---")
                    

                except Exception as e:
                    logging.error(f"Error al procesar la consulta del usuario: {e}", exc_info=True)
                    error_message = f"Lo siento, ocurrió un error al procesar tu pregunta: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Lo siento, ocurrió un error: {e}",
                        "timestamp": datetime.now(colombia_tz).isoformat()
                    })

if __name__ == "__main__":
    # --- Carga de CSS global (si no se aplica con stylable_container o st.markdown dentro de main) ---
    # html(chat_input_css) # Si es global y no se puede poner en st.markdown

    main()