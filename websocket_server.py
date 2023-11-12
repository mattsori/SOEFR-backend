import asyncio
import websockets
import json
import datetime
import os
from pathlib import Path

# Constants
SAMPLE_RATE = 48000
OVERLAP_DURATION_MS = 300
BYTES_PER_SAMPLE = 2
OVERLAP_SIZE = (SAMPLE_RATE * BYTES_PER_SAMPLE * OVERLAP_DURATION_MS) // 1000
LONG_CHUNK_AMOUNT = 5
RECORDINGS_DIR = "recordings"

# Ensure 'recordings' directory exists
Path(RECORDINGS_DIR).mkdir(parents=True, exist_ok=True)


async def save_audio(filename, audio_data):
    # Save the audio data to a file
    file_path = os.path.join(RECORDINGS_DIR, filename)
    with open(file_path, 'wb') as f:
        f.write(audio_data)
    print(f"{filename} saved.")
    # Here you would add any additional logic for handling the audio, such as transcription.


async def handler(websocket):
    print("WebSocket connection established")
    sequence = 0
    combined_chunks = bytearray()
    previous_audio_buffer = bytearray()

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Audio chunk handling logic here
                print("Audio chunk received")
                sequence += 1
                combined_chunks.extend(message)
                # Add your processing logic here
                # Save the audio chunk
                filename = f"audio_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{sequence}.wav"
                await save_audio(filename, message)
                # Send back a confirmation or any additional data required
                await websocket.send(json.dumps({'status': 'received', 'sequence': sequence}))
            else:
                # Handle non-binary message (e.g., JSON)
                print("Received message:", message)
    except websockets.exceptions.ConnectionClosed as e:
        print("WebSocket connection closed", e)

# Start the WebSocket server
start_server = websockets.serve(handler, "localhost", 8000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
