import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_response(messages, model="llama3.2:3b", temperature=0.2):
    """
    Generate a response using local Ollama Model

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: The model to use (default: "llama3.2:3b")
        temperature: Controls randomness (0.0 to 1.0, default: 0.2)

    Returns:
        The assistant's response as a string
    """
    llm = ChatOllama(model=model, temperature=temperature)
    try:
        response = llm.invoke(
            messages,
        )
        return response.content
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"
