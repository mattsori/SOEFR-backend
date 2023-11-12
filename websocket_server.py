import asyncio
import websockets
import aiohttp
import ipaddress
import json
import ssl
import datetime
import os
import wave
import speech_recognition as sr
from pathlib import Path

# Constants
WSS_PORT = 8000

SAMPLE_RATE = 48000
AUDIO_DURATION = 3
OVERLAP_DURATION_MS = 300
BYTES_PER_SAMPLE = 2
OVERLAP_SIZE = (SAMPLE_RATE * BYTES_PER_SAMPLE * OVERLAP_DURATION_MS) // 1000
LONG_AUDIO_AMOUNT = 5
RECORDINGS_DIR = "recordings"

# Ensure 'recordings' directory exists
Path(RECORDINGS_DIR).mkdir(parents=True, exist_ok=True)


ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
# Specify the path to your SSL certificate and private key
ssl_context.load_cert_chain("cloudflare-cert.pem", "cloudflare-key.pem")

# Read Cloudflare IP addresses from the JSON file
with open('cloudflare_ips.json', 'r') as json_file:
    cloudflare_ips = json.load(json_file)["cloudflare_ips"]

allowed_networks = [ipaddress.ip_network(range) for range in cloudflare_ips]

connected_clients = set()  # Maintain a set of connected clients


async def is_cloudflare_ip(ip):
    for network in allowed_networks:
        if ipaddress.ip_address(ip) in network:
            return True
    return False


async def save_audio(filename, audio_data):
    # Calculate the number of frames based on the audio data length
    num_frames = len(audio_data) // BYTES_PER_SAMPLE

    # Save the audio data to a file
    file_path = os.path.join(RECORDINGS_DIR, filename)
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)  # Assuming mono audio
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(SAMPLE_RATE)
        wf.setnframes(num_frames)
        wf.writeframes(audio_data)

    print(f"{filename} saved.")
    # Here you would add any additional logic for handling the audio, such as transcription.


async def transcribe_audio(filename, size, ws):
    if size == 'short':
        flask_server_url = 'http://localhost:8001/transcribeshort'
    else:
        flask_server_url = 'http://localhost:8002/transcribelong'
    payload = {
        'audio_file_path': os.path.join(RECORDINGS_DIR, filename),
        'audio_size': size
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(flask_server_url, json=payload) as response:
            if response.status == 200:
                print("Audio sent for transcription")
                transcription_data = await response.json()
                transcription = transcription_data.get('transcription', '')
                message = {
                    "transcript": transcription,
                    "audio_size": size
                }
                json_message = json.dumps(message)
                await ws.send(json_message)
                print("Transcription:", transcription)
            else:
                print("Failed to send audio to transcription server.",
                      response.status)


async def websocket_server(websocket):
    # Add client to set of connected clients
    connected_clients.add(websocket)

    # Check the client's IP address
    client_ip = websocket.remote_address[0]

    if await is_cloudflare_ip(client_ip):

        # Allow the connection
        await websocket.send("Connected to WebSocket server")
        print(f"{websocket.remote_address[0]} has connected")

        sequence = 0
        audio_saved = 0
        combined_chunks = bytearray()
        long_chunks = bytearray()
        total_audio_bytes = SAMPLE_RATE * BYTES_PER_SAMPLE * AUDIO_DURATION
        overlap_buffer = bytearray()

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Audio chunk handling logic here
                    combined_chunks.extend(message)
                    long_chunks.extend(message)

                    # Check if we have 3 seconds worth of audio
                    while len(combined_chunks) >= total_audio_bytes + OVERLAP_SIZE:
                        # Save the audio chunk with overlap
                        audio_data_with_overlap = overlap_buffer + \
                            combined_chunks[:total_audio_bytes + OVERLAP_SIZE]
                        filename = f"audio_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{sequence}.wav"
                        await save_audio(filename, audio_data_with_overlap)

                        # Prepare the overlap buffer for the next audio chunk
                        overlap_buffer = combined_chunks[total_audio_bytes:
                                                         total_audio_bytes + OVERLAP_SIZE]
                        # Remove the saved audio from the buffer
                        combined_chunks = combined_chunks[total_audio_bytes + OVERLAP_SIZE:]

                        sequence += 1
                        audio_saved += 1

                        await transcribe_audio(filename, 'short', websocket)

                        if audio_saved % LONG_AUDIO_AMOUNT == 0:
                            print(f"audio_saved = {audio_saved}")
                            filename = f"combinedaudio_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{sequence}.wav"
                            await save_audio(filename, long_chunks)

                            await transcribe_audio(filename, 'long', websocket)

                            long_chunks = bytearray()

                else:
                    # Handle non-binary message (e.g., JSON)
                    print("Received message:", message)
        except websockets.exceptions.ConnectionClosed as e:
            print("WebSocket connection closed", e)

if __name__ == '__main__':
    # Start the WebSocket server
    start_server = websockets.serve(
        websocket_server, '0.0.0.0', WSS_PORT, ssl=ssl_context)

    print(f"Server is running on port {WSS_PORT}")

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(start_server)
        loop.run_forever()
    except KeyboardInterrupt:
        # Handle any cleanup here before stopping the program
        print("Server is shutting down.")
    finally:
        # Cancel all future and pending tasks
        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()
        # Gather all tasks and stop the loop
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.stop()
        loop.close()
        print("Server has shut down successfully.")
