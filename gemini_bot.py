import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv


# load_dotenv("fun_load2.env") 
API_KEY = st.secrets["GOOGLE_API_KEY"]

# --- Page Configuration (must be the first Streamlit command) ---
st.set_page_config(
    page_title="Gemini Chatbot",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

def configure_api(api_key_to_configure):
    """Configures the Gemini API with the provided key."""
    try:
        genai.configure(api_key=api_key_to_configure)
        return True
    except Exception as e:
        
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
        st.error("This might be due to an invalid API key (check `fun_load2.env`), network issues, or the model not being available. "
                 "Try updating `google-generativeai` (`pip install --upgrade google-generativeai`).")
        return None


if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False
if 'attempted_initial_config' not in st.session_state: 
    st.session_state.attempted_initial_config = False

if not st.session_state.api_configured and not st.session_state.attempted_initial_config:
    if API_KEY:
        if configure_api(API_KEY):
            st.session_state.api_configured = True
            
            if "gemini_model" in st.session_state: del st.session_state.gemini_model
            if "chat_session" in st.session_state: del st.session_state.chat_session
            if "messages" in st.session_state: st.session_state.messages = []
        else:
            st.session_state.api_configured = False 
    else:
        st.session_state.api_configured = False 
    st.session_state.attempted_initial_config = True


# --- Streamlit UI ---
st.title("✨ Gemini Chatbot")
st.caption("A Streamlit interface for Google's Gemini")

# --- Sidebar for Settings ---
with st.sidebar:
    st.header("Configuration Status")
    if not API_KEY:
        st.error(
            "🔴 `GOOGLE_API_KEY` not found in `fun_load2.env`. "
            "Please create/update the file with your key and restart the app."
        )
    elif st.session_state.get('api_configured'):
        st.success("✅ API Key configured successfully!")
    else:
        st.error(
            "🔴 Failed to configure API with the key from `fun_load2.env`. "
            "Please check the key, your Google AI Studio project settings, and then restart the app."
        )

    if st.session_state.get('api_configured', False):
        st.markdown("---")
        st.subheader("Chat Settings")
        if st.button("Clear Chat History", key="clear_chat_main"):
            st.session_state.messages = []
            if "chat_session" in st.session_state:
                del st.session_state.chat_session
            st.rerun()
    else:
        st.warning("API not configured. Chat functionality is disabled.")

    st.markdown("---")
    st.markdown("Built with [Streamlit](https://streamlit.io) & [Gemini](https://ai.google.dev/).")
    st.title("✨ Made By Noman")

# --- Main Chat Logic ---
if not st.session_state.get('api_configured', False):
    st.info(
        "API not configured. Please ensure `GOOGLE_API_KEY` is correctly set in your "
        "`fun_load2.env` file and restart the Streamlit application."
    )
else:
    if "gemini_model" not in st.session_state:
        st.session_state.gemini_model = initialize_gemini_model()

    if not st.session_state.gemini_model:
        st.error(
            "Model could not be initialized. This might be due to an issue with the API key "
            "in `fun_load2.env` or network problems. Please verify and restart."
        )
        st.stop()

    if "chat_session" not in st.session_state:
        try:
            st.session_state.chat_session = st.session_state.gemini_model.start_chat(history=[])
        except Exception as e:
            st.error(f"Failed to start chat session: {e}. This could be an API key or network issue. Please check `fun_load2.env` and restart.")
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
            st.error("Chat session not available. Please ensure the API key in `fun_load2.env` is correct and restart.")
            st.stop()

        try:
            with st.spinner("Gemini is thinking..."):
                response = st.session_state.chat_session.send_message(user_prompt)

            if response.parts:
                gemini_response_content = "".join(part.text for part in response.parts)
            elif response.prompt_feedback and response.prompt_feedback.block_reason:
                gemini_response_content = f"Response blocked due to: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason.name}. Please rephrase your prompt."
                st.warning(gemini_response_content)
            else:
                if not response.candidates or not response.candidates[0].content.parts:
                     gemini_response_content = "Sorry, I received an empty response from the model. Please try again or rephrase."
                else:
                     gemini_response_content = "Sorry, I didn't get a valid response. Please try again."
                st.warning(gemini_response_content)


            st.session_state.messages.append({"role": "model", "content": gemini_response_content})
            with st.chat_message("model"):
                st.markdown(gemini_response_content)

        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            
            api_key_related_error = False
            if isinstance(e, genai.types.generation_types.BlockedPromptException):
                gemini_response_content = f"Response blocked. Reason: {e}"
            elif isinstance(e, genai.types.generation_types.StopCandidateException):
                gemini_response_content = f"Response generation stopped. Reason: {e}"
            elif "API_KEY" in str(e).upper() or "PERMISSION_DENIED" in str(e).upper() or "UNAUTHENTICATED" in str(e).upper():
                st.error(
                    "API Key issue detected (e.g., invalid, unauthenticated, or permission denied). "
                    "Please verify `GOOGLE_API_KEY` in `fun_load2.env` and restart the app."
                )
                st.session_state.api_configured = False
                if "gemini_model" in st.session_state: del st.session_state.gemini_model
                if "chat_session" in st.session_state: del st.session_state.chat_session
                gemini_response_content = f"Critical API Error. Please check logs and .env file. Error: {str(e)[:100]}..."
                api_key_related_error = True 
            elif "RESOURCE_EXHAUSTED" in str(e).upper():
                 st.error("Quota possibly exceeded. Check your Google Cloud Console or AI Studio billing and quotas.")
                 gemini_response_content = f"Error: Resource exhausted. {str(e)[:100]}..."
            else:
                gemini_response_content = f"Sorry, an unexpected error occurred: {str(e)[:100]}..."

            if not api_key_related_error: 
                st.session_state.messages.append({"role": "model", "content": gemini_response_content})
                with st.chat_message("model"):
                    st.markdown(gemini_response_content)
            
            if api_key_related_error:
                st.rerun()
