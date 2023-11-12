from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import time

app = Flask(__name__)

model_size = "large-v2"

# Run on GPU with FP16
print("Loading Faster Whisper Model")
audio_model = WhisperModel(model_size, device="cuda", compute_type="int8")
print("Model loaded")


@app.route('/transcribeshort', methods=['POST'])
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
            # Transcribe the audio file with voice activity
            segments, info = audio_model.transcribe(
                audio_file_path, beam_size=5, vad_filter=True)
            # print("Detected language '%s' with probability %f" %
            #       (info.language, info.language_probability))

            # Transcription process
            print(f"Transcribing {audio_size} audio")
            compute_start_time = time.time()
            for segment in segments:
                print("[%.2fs -> %.2fs] %s" %
                      (segment.start, segment.end, segment.text))
                transcription += segment.text
            elapsed_time = time.time() - compute_start_time

            # Print compute time
            if elapsed_time < 1:
                # Convert to milliseconds and print with up to 3 significant digits.
                print(f"Transcription took {elapsed_time * 1000:.3g} ms")
            else:
                # In seconds
                print(f"Transcription took {elapsed_time:.3g} secs")

            # return jsonify({"transcription": transcription})
            return jsonify({
                "transcription": transcription
            })

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


if __name__ == '__main__':
    app.run(port=8001)
