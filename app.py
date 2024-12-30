import streamlit as st
from streamlit_mic_recorder import mic_recorder
from langchain.memory import StreamlitChatMessageHistory
from chatbot import ChatBot
from audio_utils import *

def set_send_input():
    """Sets the send_input session state to True."""
    st.session_state.send_input = True
    clear_input_field()

def clear_input_field():
    """Clears the user input text field."""
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def main():
    # Set app title
    st.title("Banking Customer Chat System ")

    # Container for chat history
    chat_container = st.container()

    # Initialize session state variables
    if "send_input" not in st.session_state:
        st.session_state.send_input = False
        st.session_state.user_question = ""

    # Initializing chat history
    chat_history = StreamlitChatMessageHistory(key='history')

    # Initialize ChatBot instance
    chatbot = ChatBot(chat_history=chat_history)

    # Text input field for user questions/messages
    user_input = st.text_input("Type your message", key="user_input", on_change=set_send_input)

    # Column for text field and voice recording field
    voice_recording_column, send_button_column = st.columns([4, 1])

    # Voice recording field
    with voice_recording_column:
        voice_recording = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", key="voice_recording")

    with send_button_column:
        send_button = st.button("Send", key="send_button", on_click=clear_input_field)
    
    # Handling voice recording input
    if voice_recording:
        try:
            transcribed_audio = transcribe_audio(voice_recording)
            response = chatbot.get_response(transcribed_audio)
            intent = chatbot.classify_text(transcribed_audio)
            st.write("User intent for the query is: ", intent)
            text_to_speech(response)
            play_audio("response.wav")
        except Exception as e:
            st.error(f"Error during voice input handling: {e}")
    
    # Handling text input
    if send_button or st.session_state.send_input:
        if st.session_state.user_question != "":
            response = chatbot.get_response(st.session_state.user_question)
            intent = chatbot.classify_text(st.session_state.user_question)
            st.write("User intent for the query is: ", intent)
            st.session_state.user_question = ""
            text_to_speech(response)
            play_audio("response.wav")

    # Display chat history
    if chat_history.messages:
        with chat_container:
            st.write("Chat History:")
            for message in chat_history.messages:
                st.chat_message(message.type).write(message.content)

if __name__ == "__main__":
    main()
