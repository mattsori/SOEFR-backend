from dotenv import load_dotenv # type: ignore
from openai import AsyncOpenAI # type: ignore

# Load the environment variables from the .env file
load_dotenv()

# Load the OpenAI API key from environment variables
client = AsyncOpenAI()

async def generate_response(request):
    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=request
        )
        return response
    except Exception as error:
        print(f"Error: {error}")
