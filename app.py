from flask import Flask, request, jsonify
import librosa
import os
import uuid
import soundfile as sf
import runpod

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/analyze-bpm", methods=["POST"])
def analyze_bpm():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save the uploaded file
    filename = f"{uuid.uuid4()}.wav"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        # Load audio and estimate BPM
        y, sr = librosa.load(file_path, sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return jsonify({"bpm": round(float(tempo), 2)})


    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)

# RunPod Serverless entry point
runpod.serverless.start({"app": app})



