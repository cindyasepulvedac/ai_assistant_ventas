import os
import requests
from typing import Optional, Union, List

from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompt_values import StringPromptValue
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class TogetherChatLLM(Runnable):
    def __init__(self, api_key, model="meta-llama/Llama-3-8b-chat-hf", temperature=0.3):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def invoke(self, input, config=None, **kwargs):
        messages = [{"role": "system", "content": "Eres un asistente útil de ventas."}]

        if isinstance(input, list) and all(isinstance(m, BaseMessage) for m in input):
            for msg in input:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                messages.append({"role": role, "content": msg.content})

        elif isinstance(input, StringPromptValue):
            messages.append({"role": "user", "content": input.text})

        elif isinstance(input, str):
            messages.append({"role": "user", "content": input})

        else:
            raise ValueError(f"Formato de entrada no soportado para TogetherChatLLM: {type(input)}")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 1024
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def get_custom_prompt():
    PROMPT_TEMPLATE = """
    You are a friendly and helpful AI assistant that supports sales advisors from Comfama's *Venta Empresarial* unit by answering their questions based on tabular data.

    Your role is to:
    - Answer sales-related questions using the **provided context**, which comes from CSV/tabular sources.
    - Politely reply to greetings (e.g., "hola", "¿cómo estás?"), farewells (e.g., "gracias", "hasta luego"), and general small talk — even if no context is provided. Keep it brief and friendly 😊.
    
    Use emojis when appropriate to make your response feel natural and engaging. Always reply in **Spanish**.

    ---

    ## ⬇️ Context information (may be empty):
    {context}

    ## ❓ User question:
    {question}

    ---

    ### 🎯 Rules for your answer:

    1. If the user's input is a greeting, thanks, farewell, or casual comment — respond in a warm, friendly tone even **if no context is present**.
    2. If the input is a question related to *Venta Empresarial* data and context is provided, answer precisely using the context only.
    3. If the input is about sales but the context doesn't contain the answer, say: **"Lo siento, no cuento con la información suficiente para responder a tu pregunta."**
    4. If the input is unrelated to sales (e.g., temas políticos, personales, técnicos), kindly indicate that your scope is limited.
    5. Always respond in Spanish.
    6. Always use a professional and human tone, oriented to support *ventas* advisors.
    7. Examples of valid questions:
    - "¿Qué tipo de productos compra la empresa con id 3?"
    - "¿Cuáles son los arquetipos más comunes?"
    - "¿Qué le puedo ofrecer a empresas interesadas en programas de salud y cuidado de sus empleados?"


    Respuesta:
    """
    return PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])


def load_qa_chain(
    faiss_path,
    together_api_key,
    google_api_key,
    model_name = "meta-llama/Llama-3-8b-chat-hf"
):
    try:
        # Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )

        # Vector store
        vector_store = FAISS.load_local(
            faiss_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # LLM
        llm = TogetherChatLLM(api_key=together_api_key, model=model_name)

        # Prompt
        prompt = get_custom_prompt()

        # QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )
        return qa_chain

    except Exception as e:
        print(f"[ERROR] No se pudo cargar el QA chain: {e}")
        return None

#######################################################################################################
# def generate_ai_response(user_input):
#     messages = [
#                 SystemMessage(
#                     content="""You are an assistant who answers questions about biodiversity in Comfama parks, located in Antioquia, Colombia. 
#                             Users would often ask about species (animals and plants) found in this region.
#                             Users may ask about both wild and domestic species, so you must consider both to answer, not only wild organisms.

#                             Always assume that the user's questions are related to previous messages, unless stated otherwise. 
#                             If a user implicitly refers to a species already mentioned, use the conversation history to determine which one and respond accordingly, 
#                             without requiring the user to repeat the species name.

#                             You are friendly and polite. Greet the user **only if this is the first message in the conversation**. 
#                             Respond to expressions of gratitude and say goodbye when appropriate.

#                             At the end of each answer, **ask the user what else they would like to identify or know**, using a friendly tone in Spanish. Examples: "¿Cómo puedo ayudarte hoy?", "¿En qué más puedo ayudarte?".

#                             Avoid using section titles or breaking the response into parts. The output should look like a single flowing response. 
#                             Format the text using **bold**, _italics_, and punctuation when needed.

#                             Limit the description to a maximum of **500 tokens**.

#                             Only answer questions about animals, plants, biodiversity, and nature. 
#                             If another topic is mentioned, respond: "Lo siento, sólo puedo brindarte información sobre biodiversidad y naturaleza".

#                             All your responses must be written in Spanish.
#                             """
#                 ),
#                 MessagesPlaceholder(variable_name="history"),
#                 HumanMessage(
#                         content=[
#                             {
#                                 "type": "text",
#                                 "text": user_input
#                             }
#                         ]
#                     )

#             ]
    

#     return messages

# def classify_image_ai(image, img_byte_array):
#     messages = [
#                 SystemMessage(
#                     content="""You are an assistant that classifies images taken in Comfama parks (Antioquia, Colombia).

#                             Your task is to analyze the image and determine whether it shows a living organism, and if so, classify it as one of the following:

#                             - Respond with: "animal" if the image shows an animal.
#                             - Respond with: "planta" if it shows a plant.
#                             - Respond with: "hongo" if it shows a fungus.

#                             If the image does not show any organism, respond with exactly: "No organismo".

#                             Respond using only **one word or phrase** (no sentences, no lists, no additional explanation).

#                             All responses must be written in **Spanish**.
#                             """
#                 ),
#                 HumanMessage(
#                         content=[
#                             {
#                                 "type": "text",
#                                 "text": "Identify the organism in the picture."
#                             },
#                             {
#                                 "type": "image_url",
#                                 "image_url": {
#                                     "url": f"data:image/{image.format.lower()};base64,{base64.b64encode(img_byte_array).decode('utf-8')}"
#                                 }
#                             }
#                         ]
#                     )

#             ]

#     return messages

# def identify_image_ai(image, img_byte_array):
#     messages = [
#                 SystemMessage(
#                     content="""You are an assistant that identifies living organisms (animals, plants, or fungi) in images taken in Comfama parks, located in Antioquia, Colombia. 
#                             Users may send pictures of both wild and domestic species, so you must consider both when identifying organisms.

#                             Your task is to analyze the image and suggest the most likely species shown. Since you cannot be 100% sure, always express your response in terms of probability. 
#                             And give a list some feasible options of species based on the photo at first, if necessary.

#                             Start your response with a brief visual description of the organisms in the image, especially if more than one appears. Then focus on the most relevant organism.

#                             Provide a concise paragraph describing the most probable species. Indicating its **common name in bold**, preferably the one used locally. Additionally, give brief information regarding habitat, uses, and other curious facts. 
#                             Use and describe key features that help identify it (e.g. leaf shape, flower color, fruits, stem texture for plants; body shape and morphology for animals; or septa and mycelial pigmentation for fungi).
#                             If visible signs of disease or any other alteration is seen, explained concisely.

#                             Then, provide the following information in a bulleted list. Each item must be in **one line only**, formatted exactly as shown:
#                             - Nombre común
#                             - Nombre científico
#                             - Orden taxonómico
#                             - Familia
#                             - Distribución biogeográfica
#                             - Naturaleza migratoria o endemismo
#                             - Dieta
#                             - Estado de amenaza (según la Lista Roja de la UICN)
#                             - Especie invasora (Sí / No)
#                             - Certeza de identificación (accuracy percentage)

#                             Avoid using section titles or breaking the response into parts. The output should look like a single flowing response. 
#                             Format the text using **bold**, _italics_, and punctuation when needed.

#                             Limit the description to a maximum of **500 tokens**.

#                             At the end of the response, **ask the user what else they would like to identify or know**, using a friendly tone in Spanish.

#                             If the image is unrelated to biodiversity or living organisms, respond: 
#                             "Lo siento, sólo puedo identificar imágenes relacionadas con biodiversidad y naturaleza."

#                             All responses must be written in Spanish.
#                             """
#                 ),
#                 MessagesPlaceholder(variable_name="history"),
#                 HumanMessage(
#                         content=[
#                             {
#                                 "type": "text",
#                                 "text": "Identify the organism in the picture."
#                             },
#                             {
#                                 "type": "image_url",
#                                 "image_url": {
#                                     "url": f"data:image/{image.format.lower()};base64,{base64.b64encode(img_byte_array).decode('utf-8')}"
#                                 }
#                             }
#                         ]
#                     )

#             ]

#     return messages

# def extract_text_from_content(content):
#     if isinstance(content, str):
#         return content
#     elif isinstance(content, list):
#         texts = [block["text"] for block in content if block["type"] == "text"]
#         return " ".join(texts).strip()
#     return ""

# def contains_image(content):
#     if isinstance(content, list):
#         return any(block["type"] == "image_url" for block in content)
#     return False

# def build_history_messages_preserving_last_image(n=5):
#     """
#     Construye el historial de conversación conservando:
#     - Texto de los últimos `n` turnos (usuario + asistente)
#     - Solo la última imagen enviada por el usuario (si existe)

#     Returns:
#         List[BaseMessage]: Lista de HumanMessage / AIMessage con historial relevante
#     """
#     history = []
#     recent_msgs = st.session_state.messages[-n*2:]

#     # Buscar la última imagen (si existe)
#     last_image_msg = None
#     for msg in reversed(st.session_state.messages):
#         if msg["role"] == "user" and contains_image(msg["content"]):
#             last_image_msg = msg["content"]
#             break

#     for msg in recent_msgs:
#         role = msg["role"]
#         content = msg["content"]

#         if role == "user":
#             if contains_image(content):
#                 # Si es una imagen pero no es la última, omitirla
#                 if content != last_image_msg:
#                     continue
#                 # Si es la última, incluirla completa (texto + imagen)
#                 history.append(HumanMessage(content=content))
#             else:
#                 # Mensaje de texto normal
#                 text = extract_text_from_content(content)
#                 if text:
#                     history.append(HumanMessage(content=text))

#         elif role == "assistant":
#             text = extract_text_from_content(content)
#             if text:
#                 history.append(AIMessage(content=text))

#     return history


# def merge_history_with_prompt(prompt_messages, history_messages):
#     """
#     Inserta los mensajes del historial en la posición definida por `MessagesPlaceholder` 
#     dentro de una lista base de mensajes (`prompt_messages`).

#     Esta función permite reemplazar manualmente el marcador `MessagesPlaceholder` por los 
#     mensajes reales del historial, permitiendo un flujo personalizado sin usar 
#     `ChatPromptTemplate`.

#     Args:
#         prompt_messages (List[BaseMessage | MessagesPlaceholder]):
#             Lista base de mensajes, donde uno de ellos puede ser un `MessagesPlaceholder`.
        
#         history_messages (List[BaseMessage]):
#             Lista de mensajes del historial que reemplazarán el `MessagesPlaceholder`.

#     Returns:
#         List[BaseMessage]: Lista final de mensajes combinando prompt + historial.
#     """
#     final_messages = []
#     for msg in prompt_messages:
#         if isinstance(msg, MessagesPlaceholder):
#             final_messages.extend(history_messages)
#         else:
#             final_messages.append(msg)
#     return final_messages


# def create_image_user_message(image, img_byte_arr, prompt_text="Identify the organism in the picture."):
#     """
#     Crea un mensaje multimodal (texto + imagen) compatible con LangChain y OpenAI.

#     Args:
#         image (PIL.Image): Imagen enviada por el usuario.
#         img_byte_arr (bytes): Representación en bytes de la imagen.
#         prompt_text (str): Instrucción textual que acompaña la imagen.

#     Returns:
#         dict: Diccionario con 'role' y 'content' compatible con st.session_state.messages
#     """
#     multimodal_message = {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": prompt_text},
#             {"type": "image_url", "image_url": {
#                 "url": f"data:image/{image.format.lower()};base64,{base64.b64encode(img_byte_arr).decode('utf-8')}"
#                 }}
#             ]
#     }

#     return multimodal_message

# def append_chat_message(role, display_content, model_content=None):
#     """
#     Agrega un mensaje a ambas listas:
#     - `display_messages`: para interfaz visual (texto o imagen)
#     - `messages`: para historial estructurado (solo texto o multimodal)

#     Args:
#         role (str): "user" o "assistant"
#         display_content: contenido para mostrar (texto o PIL.Image)
#         model_content: contenido para LangChain (texto o bloque multimodal). Si es None, se usa display_content
#     """
#     st.session_state.display_messages.append({
#         "role": role,
#         "content": display_content
#     })

#     if model_content is None:
#         model_content = display_content

#     st.session_state.messages.append({
#         "role": role,
#         "content": model_content
#     })
