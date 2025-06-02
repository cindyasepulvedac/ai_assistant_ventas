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
    5. Always respond in Spanish. Avoid answers in english or other languages.
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
