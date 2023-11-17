from dotenv import load_dotenv
import openai
from openai import AsyncOpenAI
import asyncio

# Load the environment variables from the .env file
load_dotenv()

# Load the OpenAI API key from environment variables
client = AsyncOpenAI(
    timeout=5.0,
)

async def generate_response(request):
    try:
        # Making the request asynchronously
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=request
        )
        return response
    except openai.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except openai.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
    except openai.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)
    except Exception as error:
        print(f"Error: {error}")
