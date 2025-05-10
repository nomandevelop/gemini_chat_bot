import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv


load_dotenv("fun_load2.env") 

# --- Page Configuration (must be the first Streamlit command) ---
st.set_page_config(
    page_title="Gemini Chatbot", 
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---
def configure_api(api_key):
    """Configures the Gemini API with the provided key."""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error configuring API: {e}")
        return False

def initialize_gemini_model(model_name="models/gemini-1.5-flash-latest"):
    """Initializes the Gemini model."""
    st.write(f"Attempting to initialize model: {model_name}") 
    try:
        generation_config = {
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": 32,
            "max_output_tokens": 2048,
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        return model
    except Exception as e:
        st.error(f"Error initializing model '{model_name}': {e}")
        st.error("This might be due to an invalid API key, network issues, or the model not being available. "
                 "Try updating `google-generativeai` (`pip install --upgrade google-generativeai`).")
        return None

# --- Streamlit UI ---
st.title(" Gemini Chatbot")
st.caption("A Streamlit interface for Google's Gemini Pro")

# --- Sidebar for API Key and Settings ---
with st.sidebar:
    st.header("Configuration")
    api_key_env = os.getenv("GOOGLE_API_KEY") 
    api_key_input = st.text_input(
        "Enter your Google API Key:",
        type="password",
        value=api_key_env if api_key_env else "",
        help="Get your API key from Google AI Studio."
    )

    if not api_key_input:
        st.warning("Please enter your Google API Key to use the chatbot.")
        st.stop() 

    
    if st.button("Configure API Key") or \
       'api_configured' not in st.session_state or \
       st.session_state.get('api_key_used_for_config') != api_key_input:

        if configure_api(api_key_input):
            st.session_state.api_configured = True
            st.session_state.api_key_used_for_config = api_key_input
            st.success("API Key configured successfully!")
            
            if "gemini_model" in st.session_state: del st.session_state.gemini_model
            if "chat_session" in st.session_state: del st.session_state.chat_session
            if "messages" in st.session_state: del st.session_state.messages 
            st.rerun() 
        else:
            st.session_state.api_configured = False
            st.error("Failed to configure API Key. Please check and try again.")
            st.stop()

    if st.session_state.get('api_configured', False):
        st.markdown("---")
        st.subheader("Chat Settings")
        if st.button("Clear Chat History", key="clear_chat"):
            st.session_state.messages = []
            if "chat_session" in st.session_state:
                del st.session_state.chat_session 
            st.rerun()


        st.markdown("---")
        st.markdown("Built with [Streamlit](https://streamlit.io) & [Gemini](https://ai.google.dev/).")

# --- Main Chat Logic ---
if not st.session_state.get('api_configured', False):
    st.info("Please configure your API Key in the sidebar to start chatting.")
else:
    if "gemini_model" not in st.session_state:
        st.session_state.gemini_model = initialize_gemini_model() 
        

    if not st.session_state.gemini_model:
        st.error("Model could not be initialized. Please check API key and configuration in the sidebar.")
        st.stop()

    if "chat_session" not in st.session_state:
        try:
            st.session_state.chat_session = st.session_state.gemini_model.start_chat(history=[])
        except Exception as e:
            st.error(f"Failed to start chat session: {e}")
            st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_prompt := st.chat_input("Ask Gemini..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        
        if "chat_session" not in st.session_state: 
            st.error("Chat session not available. Please reconfigure API.")
            st.stop()

        try:
            with st.spinner("Gemini is thinking..."):
                response = st.session_state.chat_session.send_message(user_prompt)

            if response.parts:
                gemini_response_content = "".join(part.text for part in response.parts)
            elif response.prompt_feedback and response.prompt_feedback.block_reason:
                gemini_response_content = f"Response blocked due to: {response.prompt_feedback.block_reason.name}. Please rephrase your prompt."
                st.warning(gemini_response_content)
            else:
                gemini_response_content = "Sorry, I didn't get a valid response. Please try again."
                st.warning(gemini_response_content)

            st.session_state.messages.append({"role": "model", "content": gemini_response_content})
            with st.chat_message("model"):
                st.markdown(gemini_response_content)

        except Exception as e:
            error_message = f"An error occurred while sending message: {e}"
            st.error(error_message)
            if "API_KEY_INVALID" in str(e) or "PERMISSION_DENIED" in str(e):
                st.error("API Key issue. Please verify it in the sidebar.")
                st.session_state.api_configured = False
                if "gemini_model" in st.session_state: del st.session_state.gemini_model
                st.rerun()
            elif "RESOURCE_EXHAUSTED" in str(e):
                 st.error("Quota possibly exceeded. Check Google Cloud Console or AI Studio.")
            # Add error to chat display
            st.session_state.messages.append({"role": "model", "content": f"Sorry, error: {str(e)[:100]}..."})
            with st.chat_message("model"):
                st.markdown(f"Sorry, error: {str(e)[:100]}...")