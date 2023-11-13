import os
import openai
import asyncio

# Load the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

async def generate_response(request):
    try:
        # Making the request asynchronously
        response = await openai.ChatCompletion.create_async(
            model="gpt-3.5-turbo",
            messages=request
        )
        ai_response = response.get('choices')[0].get('message')
        print(response)  # Or handle the response as needed
        return ai_response
    except Exception as error:
        print(f"Error: {error}")

# If you want to use this as a module and not run the async function immediately,
# you can define another function that creates and runs the event loop:
def run_generate_response(request):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If the loop is already running (like in Jupyter notebook, or an existing server), we use run_until_complete
        return loop.run_until_complete(generate_response(request))
    else:
        # Otherwise, we create a new loop and close it after we're done
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(generate_response(request))
        loop.close()
        return result
