import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load the environment variables from the .env file
load_dotenv()

# Load the OpenAI API key from environment variables
client = AsyncOpenAI ()

async def generate_response(request):
    try:
        # Making the request asynchronously
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=request
        )
        ai_response = response.get('choices')[0].get('message')
        print(response)  # Or handle the response as needed
        return ai_response
    except Exception as error:
        print(f"Error: {error}")
