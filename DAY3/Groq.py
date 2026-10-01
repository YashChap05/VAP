import json
import os
import subprocess

import psutil
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL_NAME = "llama-3.3-70b-versatile"

print("=== YOUR ACTIVE GROQ MODELS ===")
for model in client.models.list().data:
    print(f"- {model.id}")   
    
    
def echo(text: str) -> str:
    """Return the given text unchanged.
    Args:
        text: The text to echo back.
    """
    return text

TOOLS = [echo]
SYSTEM_PROMPT = """
You are an echo agent. You will receive a message and you will return it unchanged.
"""