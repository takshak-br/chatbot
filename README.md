# Banking Customer Support System

This repository implements a chatbot system for banking customer support, leveraging Streamlit for the user interface, sentence transformers for intent classification, and Gemini LLM for response generation with contextual understanding.

## Key Features:

- **Speech Recognition:** Handles user audio input using OpenAI Whisper for efficient transcription.
- **Intent Classification:** Classifies user queries into predefined intents based on sentence embeddings and cosine similarity.
- **Contextual Response Generation:** Utilizes Gemini LLM with chat history memory to provide informative and relevant responses.
- **Text-to-Speech Output:** Converts chatbot responses to audio for natural interaction.

## Installation:

1. Clone this repository.
2. Create a virtual environment (recommended).
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set the OpenAI API key in a .env file:

   ```bash
   OPENAI_API_KEY="your_openai_api_key"
   ```

## Usage:

1. Run the application:

   ```bash
   streamlit run app.py
   ```

2. Interact with the chatbot through the Streamlit interface:

- Type your question or message in the text field.
- Alternatively, click the microphone icon and record your audio query.

## Code Structure:

- app.py: Main application logic, managing Streamlit UI elements, handling user input, and coordinating chatbot interactions.
- chatbot.py: Defines the ChatBot class, responsible for response generation using LLMChain and intent classification based on sentence transformers.
- audio_utils.py: Provides functions for audio transcription (using OpenAI Whisper), text-to-speech conversion (using gTTS), and audio playback in the Streamlit app.

## Technical Details:

### Intent Classification:

Sentence embeddings are generated using a pre-trained sentence transformer model for user input and predefined intents. Cosine similarity is calculated between the user input embedding and all dataset embeddings to identify the most similar intent.

### Chatbot Response Generation:

A ChatBot instance is created, incorporating an LLMChain object with a customized prompt template. The prompt template leverages the chat history memory to provide context-aware responses. The user's query and chat history are fed into the LLMChain to generate a response aligned with the classified intent.

### Audio Processing:

OpenAI Whisper is used for transcribing speech input to text. gTTS library handles the text-to-speech conversion, saving speech output as a WAV file. Streamlit displays transcribed text and plays the generated speech file.
