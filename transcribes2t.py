from flask import Flask, request, jsonify
import time
import whisper_s2t

WHISPER_MODEL = "medium.en"

app = Flask(__name__)

model = whisper_s2t.load_model(WHISPER_MODEL, device="cuda", compute_type="int8", asr_options={'word_timestamps': True})
print(f"{WHISPER_MODEL} loaded")

def is_transcription_valid(transcription, audio_size):
    words = transcription.split()
    if audio_size == 'short':
        size = 3
    else:
        size = 15

    # Re-transcribe if too many words
    if len(words) > size*7:
        print(f"Too many words")
        return False

    # Count occurrences of each word
    word_count = {}
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

    # Hallucination check: Checks for any word repeated too many times
    for word, count in word_count.items():
        if count > size*2:
            print(f"Word '{word}' repeated {count} times, which is too many")
            return False

    return True  # No issues found with repetition

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    audio_file_path = [request.json['audio_file_path']]
    audio_size = request.json['audio_size']
    transcription = ""
    max_retries = 3  # Set the maximum number of retries
    retries = 0
    last_exception = None  # Define a variable to store the last exception outside of the loop
    print("Attempting to transcribe " + str(audio_file_path))
    while retries < max_retries:
        try:
            lang_codes = ['en']
            tasks = ['transcribe']
            initial_prompts = [None]

            print(f"Transcribing {audio_size} audio")
            compute_start_time = time.time()
            out = model.transcribe_with_vad(audio_file_path,
                                            lang_codes=lang_codes,
                                            tasks=tasks,
                                            initial_prompts=initial_prompts,
                                            batch_size=24)

            transcription = out[0][0]['text']
            print(transcription)
            # Transcription process

            elapsed_time = time.time() - compute_start_time

            # Print compute time
            if elapsed_time < 1:
                # Convert to milliseconds and print with up to 3 significant digits.
                print(f"Transcription took {elapsed_time * 1000:.3g} ms")
            else:
                # In seconds
                print(f"Transcription took {elapsed_time:.3g} secs")

            # Check if the transcription is valid
            if is_transcription_valid(transcription, audio_size):
                return jsonify({"transcription": transcription})
            else:
                print("Invalid transcription detected, retrying...")
                transcription = ""  # Reset transcription for a retry
                retries += 1
                # Adjust model parameters for retry or use a different strategy
                continue

        except (RuntimeError, FileNotFoundError) as e:
            print(f"Error during transcription: {e}")
            last_exception = e  # Store the exception
            retries += 1
            print(f"Retrying transcription ({retries}/{max_retries})...")
            time.sleep(1)  # Sleep before retrying

    # If retries have been exhausted, return an error response
    if last_exception:
        print("Maximum retries reached. Transcription failed.")
        return jsonify({
            "error": "Maximum retries reached. An error occurred during transcription",
            "details": str(last_exception)
        }), 500
    else:
        return jsonify({
            "error": "Unknown error occurred during transcription"
        }), 500

def start_transcribe():
    app.run(port=8001, use_reloader=False)

if __name__ == '__main__':
    app.run(port=8001)
