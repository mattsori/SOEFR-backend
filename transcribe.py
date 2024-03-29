from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import time

app = Flask(__name__)

model_size = "large-v3"

# Run on GPU with FP16
print("Loading Faster Whisper Model")
audio_model = WhisperModel(model_size, device="cuda", compute_type="int8")
print("Model loaded")

def is_transcription_valid(transcription, audio_size):
    print(f"Checking if transcription is valid")
    # Split transcription into words or phrases
    words = transcription.split()

    # Re-transcribe if too many words
    if len(words) > 20:
        print(f"Too many words")
        return False

    # Count occurrences of each word
    word_count = {}
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

    # Check for any word repeated more than three times
    for word, count in word_count.items():
        if count > 5:
            print(f"Word '{word}' repeated {count} times, which is too many")
            return False

    return True  # No issues found with repetition

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    audio_file_path = request.json['audio_file_path']
    audio_size = request.json['audio_size']

    transcription = ""
    max_retries = 3  # Set the maximum number of retries
    retries = 0
    last_exception = None  # Define a variable to store the last exception outside of the loop
    print("Attempting to transcribe " + str(audio_file_path))
    while retries < max_retries:
        try:
            print(f"Transcribing {audio_size} audio")
            compute_start_time = time.time()
            
            # Transcribe the audio file with voice activity
            segments, info = audio_model.transcribe(
                audio_file_path, beam_size=5, vad_filter=True, word_timestamps=True, temperature=0)
            # print("Detected language '%s' with probability %f" %
            #       (info.language, info.language_probability))

            # Transcription process

            for segment in segments:
                print("[%.2fs -> %.2fs] %s" %
                      (segment.start, segment.end, segment.text))
                for word in segment.words:
                    print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
                transcription += segment.text
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
