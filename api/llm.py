import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(messages, model="gpt-4.1-mini", temperature=0.2):
    """
    Generate a response from the OpenAI API
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: The model to use (default: "gpt-4.1-mini")
        temperature: Controls randomness (0.0 to 1.0, default: 0.2)
    
    Returns:
        The assistant's response as a string
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"