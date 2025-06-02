import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import io
import base64
from PIL import Image
from typing import List, Dict

class ImageAnalysisAssistant:
    def __init__(self):
        self.system_prompt = """You are an expert computer vision analyst with the following capabilities and responsibilities:

1. Image Analysis Expertise:
   - Detailed visual description
   - Object detection and recognition
   - Text extraction and interpretation
   - Scene analysis and context understanding
   - Color and composition analysis
   - Emotional and aesthetic interpretation

2. Communication Guidelines:
   - Provide structured, clear responses
   - Start with the most relevant information
   - Use bullet points for listing multiple elements
   - Include technical details when relevant
   - Maintain a professional yet accessible tone

3. Analysis Framework:
   - First: Provide a brief overview of the main elements
   - Second: Detail specific elements based on the analysis type requested
   - Third: Add relevant context or interpretations
   - Finally: Note any limitations or uncertainties in the analysis

4. Response Format:
   - Overview: [Brief summary]
   - Main Elements: [Detailed analysis]
   - Context: [Additional insights]
   - Notes: [Any important considerations]

Remember to be precise, objective, and thorough in your analysis while maintaining clarity in your explanations."""

    def get_analysis_types(self) -> Dict[str, str]:
        return {
            "General Description": "Provide a comprehensive description of all visual elements in this image, including composition, colors, and overall impression.",
            "Object Detection": "Identify and describe all distinct objects in the image. For each object, note its position, appearance, and relative importance in the scene.",
            "Text Extraction": "Extract and interpret any visible text in the image. Include context about the text's placement and appearance.",
            "Scene Analysis": "Analyze the overall scene including setting, mood, lighting, and any activities or interactions. Provide context about the likely location and time.",
            "Technical Analysis": "Evaluate technical aspects like image quality, lighting conditions, composition techniques, and any notable photographic elements.",
            "Emotional Impact": "Assess the emotional qualities and potential impact of the image, including mood, atmosphere, and intended message."
        }

    def create_messages(self, analysis_type: str, image_base64: str) -> List:
        return [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": self.get_analysis_types()[analysis_type]
                },
                {
                    "type": "image_url",
                    "image_url": image_base64
                }
            ])
        ]

def encode_image_to_base64(image_pil):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image_pil.save(buffered, format=image_pil.format if image_pil.format else 'JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/{image_pil.format.lower() if image_pil.format else 'jpeg'};base64,{img_str}"

def main():
    st.title("Advanced Image Analysis with Azure OpenAI")
    
    # Initialize the assistant
    assistant = ImageAnalysisAssistant()
    
    # Azure OpenAI Configuration in sidebar
    with st.sidebar:
        st.subheader("Azure OpenAI Configuration")
        azure_endpoint = st.text_input("Azure Endpoint URL", placeholder="https://your-resource.openai.azure.com/")
        api_key = st.text_input("Azure API Key", type="password")
        deployment_name = st.text_input("Deployment Name", placeholder="your-deployment-name")
        api_version = st.text_input("API Version", value="2024-02-15-preview")
        
        # Advanced settings collapsible
        with st.expander("Advanced Settings"):
            temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
            max_tokens = st.slider("Max Tokens", 100, 1000, 300, 50)
    
    # File uploader allows user to add their own image
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Add analysis options
        analysis_type = st.selectbox(
            "Choose analysis type",
            list(assistant.get_analysis_types().keys())
        )
        
        # Display the specific prompt for the selected analysis type
        with st.expander("View Analysis Prompt"):
            st.write(assistant.get_analysis_types()[analysis_type])
        
        # Add a button to trigger analysis
        if st.button("Analyze Image"):
            if not all([azure_endpoint, api_key, deployment_name]):
                st.error("Please enter all Azure OpenAI configuration details!")
                return
            
            try:
                # Initialize Azure OpenAI chat model
                chat = AzureChatOpenAI(
                    azure_endpoint=azure_endpoint,
                    openai_api_key=api_key,
                    deployment_name=deployment_name,
                    api_version=api_version,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Prepare the image
                base64_image = encode_image_to_base64(image)
                
                # Create messages with system prompt and user input
                messages = assistant.create_messages(analysis_type, base64_image)
                
                # Get response from the model
                with st.spinner("Analyzing image..."):
                    response = chat.invoke(messages)
                
                # Display results in a structured format
                st.subheader("Analysis Results:")
                st.markdown(response.content)
                
                # Add option to download results
                if st.button("Download Analysis"):
                    analysis_text = f"""
                    Image Analysis Report
                    ===================
                    Type: {analysis_type}
                    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    
                    Analysis Results:
                    ----------------
                    {response.content}
                    """
                    st.download_button(
                        label="Download Report",
                        data=analysis_text,
                        file_name=f"image_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()