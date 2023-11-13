import asyncio
import websockets
import aiohttp
import ipaddress
import json
import ssl
import datetime
import os
import wave
import webrtcvad
from openai_client import generate_response
from pathlib import Path

# Constants
WSS_PORT = 8000  # The WebSocket server port
CHANNEL_WIDTH = 1  # Mono audio channel
SAMPLE_RATE = 48000  # Sample rate for audio in Hz
AUDIO_DURATION = 3  # Duration of audio in seconds to process at once
BYTES_PER_SAMPLE = 2  # Number of bytes per sample in the audio
LONG_AUDIO_AMOUNT = 2  # Number of audio pieces to combine for long audio
RECORDINGS_DIR = "recordings"  # Directory to save recordings
MAX_SPEECH_LENGTH = SAMPLE_RATE * BYTES_PER_SAMPLE * AUDIO_DURATION  # Max length of speech to process
MIN_SPEECH_LENGTH = SAMPLE_RATE * BYTES_PER_SAMPLE  # Minimum speech length in bytes (1 second)
PHRASE_TIMEOUT_MS = 300  # Timeout after speech ends, in ms
FRAME_DURATION_MS = 30  # Duration of an audio frame in ms
FRAME_SIZE = (SAMPLE_RATE * FRAME_DURATION_MS * BYTES_PER_SAMPLE * CHANNEL_WIDTH) // 1000  # Size of an audio frame in bytes
Path(RECORDINGS_DIR).mkdir(parents=True, exist_ok=True)  # Ensure the recordings directory exists


# Initialize VAD
vad = webrtcvad.Vad(1)  # Aggressiveness level is defaulted to 1

# SSL context for securing WebSocket connection
ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain("cloudflare-cert.pem", "cloudflare-key.pem")

# Load the Cloudflare IP ranges to only allow connections from them
with open('cloudflare_ips.json', 'r') as json_file:
    cloudflare_ips = json.load(json_file)["cloudflare_ips"]
allowed_networks = [ipaddress.ip_network(range) for range in cloudflare_ips]
connected_clients = set()   # Keep track of connected clients


class ConnectionHandler:
    def __init__(self):
        self.speech_buffer = bytearray()  # Buffer to hold incoming audio data
        self.sequence = 0  # Sequence number for file naming
        self.audio_saved = 0  # Counter for saved audio files
        self.combined_chunks = bytearray()  # Buffer to combine audio chunks
        self.long_chunks = bytearray()  # Buffer to hold longer audio for transcription
        self.speech_segment_buffer = bytearray()  # Buffer to hold the current speech segment
        self.silence_duration_ms = 0  # Counter for the duration of silence
        self.long_transcriptions = {}  # Dictionary to store long transcriptions for each client

    # Process incoming WebSocket message
    async def process_message(self, websocket, message):
        # Binary message handling (audio data)
        if isinstance(message, bytes):
            await self.process_audio_frame(websocket, message)
        else:
            # Handle non-binary message (JSON)
            print("Received message:", message)

    # Handle incoming audio frames
    async def process_audio_frame(self, websocket, audio_frame):
        # Append new audio data to the speech buffer
        self.speech_buffer.extend(audio_frame)
        frame_number = 0
        # Process audio frames as long as there's enough data in the buffer
        while len(self.speech_buffer) >= FRAME_SIZE:
            frame_number += 1
            frame = self.speech_buffer[:FRAME_SIZE]
            self.speech_buffer = self.speech_buffer[FRAME_SIZE:]

            # Use VAD to check if the current frame contains speech
            if vad.is_speech(frame, SAMPLE_RATE):
                self.speech_segment_buffer.extend(frame)
                self.silence_duration_ms = 0  # Reset silence duration when speech is detected
            else:
                # Handle non-speech periods
                await self.handle_non_speech_periods(websocket)

    # Process non-speech periods to determine if a speech segment has ended
    async def handle_non_speech_periods(self, websocket):
        # If there's speech data in the buffer, increment the silence duration
        if len(self.speech_segment_buffer) > 0:
            self.update_silence_duration()
            # Check if silence has exceeded the timeout or the speech is too long
            if self.should_save_speech_segment():
                print("Processing speech segments")
                await self.process_speech_segment(websocket)

    def update_silence_duration(self):
        self.silence_duration_ms += FRAME_DURATION_MS

    def should_save_speech_segment(self):
        return (self.silence_duration_ms >= PHRASE_TIMEOUT_MS or len(self.speech_segment_buffer) + len(self.combined_chunks) >= MAX_SPEECH_LENGTH)

    # Process speech segments when buffer reaches required length
    async def process_speech_segment(self, websocket):
        self.combined_chunks.extend(self.speech_segment_buffer)
        self.speech_segment_buffer = bytearray()   # Clear speech segment buffer
        self.silence_duration_ms = 0  # Reset silence duration

        # Check if we have enough audio to save and transcribe
        if len(self.combined_chunks) >= MIN_SPEECH_LENGTH:
            # Construct filename and save audio
            filename = f"audio_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{self.sequence}.wav"
            await self.save_audio(filename, self.combined_chunks)
            self.long_chunks.extend(self.combined_chunks)

            # Reset combined chunks buffer
            self.combined_chunks = bytearray()
            self.sequence += 1
            self.audio_saved += 1

            # Transcribe the short audio segment
            await self.transcribe_audio(filename, 'short', websocket)

            # Every LONG_AUDIO_AMOUNT of audio pieces, transcribe long audio
            if self.audio_saved % LONG_AUDIO_AMOUNT == 0:
                filename = f"combinedaudio_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{self.sequence}.wav"
                await self.save_audio(filename, self.long_chunks)

                # Transcribe the long audio segment
                transcription = await self.transcribe_audio(filename, 'long', websocket)
                await self.save_long_transcription(transcription, websocket)
                await self.summarize(transcription)
                # Clear the long chunks buffer
                self.long_chunks = bytearray()

    # Save audio data to a .wav file
    async def save_audio(self, filename, audio_data):
        num_frames = len(audio_data) // BYTES_PER_SAMPLE
        file_path = os.path.join(RECORDINGS_DIR, filename)
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(CHANNEL_WIDTH)
            wf.setsampwidth(BYTES_PER_SAMPLE)
            wf.setframerate(SAMPLE_RATE)
            wf.setnframes(num_frames)
            wf.writeframes(audio_data)
        print(f"{filename} saved.")

    # Send audio data to transcription service and handle the response
    async def transcribe_audio(self, filename, size, ws):
        flask_server_url = f'http://localhost:800{1 if size == "short" else 2}/transcribe{size}'
        payload = {'audio_file_path': os.path.join(
            RECORDINGS_DIR, filename), 'audio_size': size}

        # Post audio file to the transcription server
        async with aiohttp.ClientSession() as session:
            async with session.post(flask_server_url, json=payload) as response:
                if response.status != 200:
                    print("Failed to send audio to transcription server.",
                          response.status)
                    return

                print("Audio sent for transcription")
                # Extract transcription from response
                transcription_data = await response.json()
                transcription = transcription_data.get('transcription', '')

                # If transcription is empty, no speech was detected
                if not transcription:
                    print("Audio contains no speech")
                    return

                # Send transcription back to the client
                message = {"transcript": transcription, "audio_size": size}
                await ws.send(json.dumps(message))
                print("Transcription:", transcription)

    # Save the long transcription for the client
    async def save_long_transcription(self, transcription, websocket):
        # Use the id() of the websocket as a unique identifier for the client
        client_id = id(websocket)
        if client_id not in self.long_transcriptions:
            self.long_transcriptions[client_id] = []
        self.long_transcriptions[client_id].append(transcription)

    # Method to get long transcriptions for a specific client
    def get_long_transcriptions_for_client(self, websocket):
        client_id = id(websocket)
        return self.long_transcriptions.get(client_id, [])
    
    async def summarize(self, request):
        instructions = "Summarize the transcription below into bullet points: "
        # Format request for OpenAI API
        formattedRequest = {
            "role": "user",
            "content": f'{instructions} + {request}'
        }
        response = await generate_response(formattedRequest)
        print(response)




# Check if an IP address is within the allowed Cloudflare range
async def is_cloudflare_ip(ip):
    return any(ipaddress.ip_address(ip) in network for network in allowed_networks)


async def websocket_server(websocket, path):
    # Initialize the handler for this connection
    handler = ConnectionHandler()
    connected_clients.add(websocket)

    # Get the IP address of the client
    client_ip = websocket.remote_address[0]
    # Only allow connections from IPs within the Cloudflare range
    if not await is_cloudflare_ip(client_ip):
        print(f"Rejected connection from {client_ip}")
        return

    await websocket.send("Connected to WebSocket server")
    print(f"{client_ip} has connected")

    try:
        async for message in websocket:
            # Handle message using the connection handler's state
            await handler.process_message(websocket, message)
    except websockets.exceptions.ConnectionClosed as e:
        # Handle connection closed events
        print("WebSocket connection closed", e)
    finally:
        # Remove the client from the connected set on disconnection
        print(f"{client_ip} has disconnected")
        connected_clients.remove(websocket)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    # Start the server and await the server to start properly
    start_server = websockets.serve(websocket_server, '0.0.0.0', WSS_PORT, ssl=ssl_context)
    print(f"Server is running on port {WSS_PORT}")
    server = loop.run_until_complete(start_server)

    try:
        # Run the event loop forever until interrupted
        loop.run_forever()
    except KeyboardInterrupt:
        print("Server is shutting down.")
    finally:
        # Stop server and wait until it is closed
        server.close()
        loop.run_until_complete(server.wait_closed())

        # Gather all pending tasks and cancel them
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        # Wait until all tasks are cancelled.
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

        # Finally close the event loop
        loop.close()
        print("Server has shutdown successfully")
