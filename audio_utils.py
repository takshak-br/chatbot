import openai
import streamlit as st
from gtts import gTTS
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API Key
openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_api_key


def transcribe_audio(audio):
    """
    Transcribes audio to text using OpenAI Whisper API.
    
    Args:
        audio (dict): Dictionary containing audio data.
    """
    # Converting audio data to a BytesIO object
    audio_bio = io.BytesIO(audio['bytes'])
    try:
        # Use the openai API to transcribe the audio
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_bio,
            language='en'
        )
        output = transcript['text']
        st.write(f"Transcription: {output}")
        return output
    except openai.error.OpenAIError as e:
        st.write(f"Error: {e}")
        return None


def text_to_speech(response):
    """
    Converts text to speech and saves it as a WAV file using gTTS library.
    
    Args:
        response (str): Text to be converted to speech.
    """
    try:
        tts = gTTS(text=response, lang='en')
        tts.save('response.wav')
    except Exception as e:
        st.write(f"An error occurred: {e}")


def play_audio(file_name):
    """
    Plays audio file in the Streamlit app.
    
    Args:
        file_name (str): Name of the audio file.
    """
    try:
        with open(file_name, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/wav")
    except Exception as e:
        st.write(f"Error while playing audio: {e}")


# Streamlit app UI
def main():
    st.title("Banking Customer Chat System")

    # Audio upload
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")  # Preview the uploaded audio

        # Convert audio to text
        transcript = transcribe_audio({'bytes': audio_file.getvalue()})

        if transcript:
            # Use the transcript as input for the chatbot (assuming you have a chatbot method)
            response = f"Here's what you said: {transcript}"

            # Convert response to speech
            text_to_speech(response)

            # Play the generated response
            play_audio('response.wav')


if __name__ == "__main__":
    main()
