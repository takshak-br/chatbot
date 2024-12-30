import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\Anusha H\\Downloads\\Banking-customer-support-chatbot-main\\api.txt"


def get_sentence_transformer():
    """Returns a SentenceTransformer model"""
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    return model

def get_llm():
    """Returns a Gemini LLM model"""
    api_key = os.getenv("GOOGLE_API_KEY")  # Use environment variable for API key
    genai.configure(api_key=api_key)
    llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.6, convert_system_message_to_human=True, google_api_key=api_key)
    return llm

def create_chat_memory(chat_history):
    """Returns a ConversationBufferWindowMemory object with a chat history of 6 messages."""
    return ConversationBufferWindowMemory(memory_key="history", chat_memory=chat_history, k=6)

def get_llm_chain(llm, memory):
    """Returns a LLMChain object with a template for a virtual assistant for banking customer support."""
    template = """
    Act as a virtual assistant for banking customer support sector. Your name is Sayogi.
    Generate a helpful and informative response that addresses their banking needs while maintaining a friendly and professional tone.
    Try to respond with a short answer using simple language without technical terms in the response.

    Make use of the previous conversation history to understand the intent of the user and answer based on the user intent.
    Previous conversation: {history}
    Human: {query}
    AI: 
    """
    prompt = PromptTemplate(template=template, input_variables=['query', 'history'])
    chain = LLMChain(prompt=prompt, llm=llm, memory=memory)
    return chain

class ChatBot:
    """A chatbot that uses a LLMChain to generate responses to user input."""
    def __init__(self, chat_history):
        self.memory = create_chat_memory(chat_history)
        self.llm = get_llm()
        self.chain = get_llm_chain(self.llm, self.memory)
        self.sentence_transformer_model = get_sentence_transformer()
        
        try:
            self.trained_embeddings = pd.read_csv('intent_embeddings.csv')
            self.dataset = pd.read_csv('intent.csv')
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            self.trained_embeddings = pd.DataFrame()  # Handle gracefully
            self.dataset = pd.DataFrame()

    def get_response(self, user_input):
        """Generates a response to user input using the LLMChain object."""
        try:
            response = self.chain.run(query=user_input, history=self.memory.chat_memory.messages)
        except Exception as e:
            print(f"Error during LLM response generation: {e}")
            response = "I'm sorry, I couldn't process your request at the moment."
        return response

    def classify_text(self, user_input):
        """Classifies user intent based on cosine similarity."""
        if self.trained_embeddings.empty or self.dataset.empty:
            return "unknown_intent"

        # Get user input embedding
        user_input_embedding = self.sentence_transformer_model.encode(user_input)

        # Calculate cosine similarity between user input and all dataset texts
        cosine_scores = cosine_similarity(user_input_embedding.reshape(1, -1), self.trained_embeddings)

        # Find the most similar text in the dataset
        most_similar_idx = np.argmax(cosine_scores)
        predicted_intent = self.dataset["intents"].iloc[most_similar_idx]
        return predicted_intent

