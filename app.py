import os
from dotenv import load_dotenv # Add this line
import sys # Import sys for stderr printing

load_dotenv() # Load environment variables from .env file

import boto3
from flask import Flask, request, jsonify
import librosa

import uuid
import soundfile as sf
import runpod # Import the runpod library

app = Flask(__name__)

# --- Configuration (IMPORTANT for Deployment) ---
# It's best practice to get sensitive info from environment variables
# For local testing, you can set these in your terminal or .env file
S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1') # Default region

# Initialize S3 client (ensure credentials are available in the environment
# or via an IAM role when deployed on RunPod)
s3_client = None # Initialize to None
if S3_BUCKET: # Only initialize if bucket name is provided
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        print(f"INFO: S3 client initialized successfully for bucket '{S3_BUCKET}' in region '{AWS_REGION}'.")
    except Exception as e:
        print(f"ERROR: Failed to initialize S3 client: {e}", file=sys.stderr)
        # S3_BUCKET is set but client failed to initialize, keep s3_client as None
else:
    print("WARNING: S3_BUCKET_NAME environment variable not set. S3 functionality will not work.", file=sys.stderr)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"INFO: Uploads directory '{UPLOAD_FOLDER}' ensured.")


# Define the core logic of your application as a standalone function
# This makes it reusable by both Flask (for local testing) and RunPod's handler
def process_audio_for_bpm(s3_file_path):
    print(f"INFO: Starting process_audio_for_bpm for path: {s3_file_path}")
    if not S3_BUCKET or not s3_client:
        print("ERROR: S3 bucket name or client not configured.", file=sys.stderr)
        raise ValueError("S3 bucket or client not configured.")

    filename = os.path.basename(s3_file_path)
    name_only = os.path.splitext(os.path.basename(s3_file_path))[0]
    local_file_path = os.path.join(UPLOAD_FOLDER, filename)

    # ✅ Move this here — before the try block so it's always defined
    temp_files = [local_file_path]

    try:
        # Step 1: Download audio from S3
        s3_client.download_file(S3_BUCKET, s3_file_path, local_file_path)
        print(f"INFO: Downloaded '{filename}' to '{local_file_path}'")

        # Step 2: Load audio
        y, sr = librosa.load(local_file_path, sr=None, mono=True)
        duration = len(y) / sr

        # Step 3: Estimate BPM
        tempo = librosa.beat.tempo(y=y, sr=sr)
        tempo = tempo[0] if isinstance(tempo, np.ndarray) else tempo
        if tempo > 100:
            tempo /= 2
        tempo = round(float(tempo), 2)

        # Step 4: Generate note time arrays
        def create_time_grid(start, end, divisions_per_measure=4):
            beat_len = 60.0 / tempo
            measure_len = beat_len * 4
            total_measures = (end - start) / measure_len
            times = []
            for m in range(int(total_measures) + 1):
                for div in range(divisions_per_measure):
                    t = start + m * measure_len + (div * measure_len / divisions_per_measure)
                    if t <= end:
                        times.append(round(t, 4))
            return np.array(times)

        whole_notes = create_time_grid(0, duration, 1)
        half_notes = create_time_grid(0, duration, 2)
        quarter_notes = create_time_grid(0, duration, 4)
        eighth_notes = create_time_grid(0, duration, 8)
        sixteenth_notes = create_time_grid(0, duration, 16)

        # Step 5: Tick overlay function
        def add_ticks(audio, times, note_type, volume=1.0):
            freq = 800
            duration_map = {
                'whole': 0.15, 'half': 0.12,
                'quarter': 0.1, 'eighth': 0.08, 'sixteenth': 0.06
            }
            tick_duration = int(duration_map[note_type] * sr)
            ticks = np.zeros_like(audio)
            for t in times:
                start = int(t * sr)
                end = min(start + tick_duration, len(audio))
                tick = volume * np.sin(2 * np.pi * freq * np.arange(end - start) / sr)
                ticks[start:end] = tick * np.linspace(1, 0, end - start)
            return audio + ticks

        # Step 6: Create and save ticked audio
        file_map = {}
        time_map = {
            "whole_notes": ("whole", whole_notes),
            "half_notes": ("half", half_notes),
            "quarter_notes": ("quarter", quarter_notes),
            "eighth_notes": ("eighth", eighth_notes),
            "sixteenth_notes": ("sixteenth", sixteenth_notes)
        }

        for key, (note_type, times) in time_map.items():
            out_filename = f"{name_only}_{key}.wav"
            out_path = os.path.join(UPLOAD_FOLDER, out_filename)
            ticked_audio = add_ticks(y.copy(), times, note_type)
            sf.write(out_path, ticked_audio, sr)
            file_map[key] = out_filename
            temp_files.append(out_path)

        print("INFO: BPM and tick sounds processing complete.")

        return {
            "bpm": tempo,
            "note_sounds": file_map,
            "note_timings": {
                "whole_notes": whole_notes.tolist(),
                "half_notes": half_notes.tolist(),
                "quarter_notes": quarter_notes.tolist(),
                "eighth_notes": eighth_notes.tolist(),
                "sixteenth_notes": sixteenth_notes.tolist()
            }
        }

    except Exception as e:
        print(f"ERROR: Exception during audio processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise e

    finally:
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"INFO: Deleted temporary file: {f}")
            except Exception as cleanup_err:
                print(f"WARNING: Failed to delete {f}: {cleanup_err}", file=sys.stderr)



# --- Flask Route for Local Development/Testing (Optional) ---
# This allows you to test your Flask app locally before deploying to RunPod.
# For production deployment on RunPod, this route won't be directly hit,
# but the 'handler' function below will use the 'process_audio_for_bpm' logic.
@app.route("/analyze-bpm", methods=["POST"])
def analyze_bpm_flask_route():
    print("INFO: Flask /analyze-bpm route hit.")
    if not request.is_json:
        print("WARNING: Flask request is not JSON.", file=sys.stderr)
        return jsonify({"error": "Request must be JSON"}), 400

    if "s3_file_path" not in request.json:
        print("WARNING: No 's3_file_path' provided in Flask request JSON.", file=sys.stderr)
        return jsonify({"error": "No 's3_file_path' provided in JSON"}), 400


    
    s3_file_path = request.json["s3_file_path"]
    print(f"INFO: Flask received s3_file_path: {s3_file_path}")

    try:
        result = process_audio_for_bpm(s3_file_path)
        print(f"INFO: Flask returning successful BPM result: {result}")
        return jsonify(result)
    except ValueError as ve:
        print(f"ERROR: Flask ValueError: {ve}", file=sys.stderr)
        return jsonify({"error": str(ve)}), 500
    except Exception as e:
        print(f"ERROR: Flask Internal server error: {e}", file=sys.stderr)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# --- RunPod Serverless Handler (REQUIRED for Deployment) ---
# This function is the entry point for your RunPod Serverless worker.
# It receives a 'job' dictionary containing the input data.
def handler(job):
    print("INFO: RunPod Serverless handler function started.")
    # The input to your API will be in job['input']
    input_data = job.get('input', {})
    print(f"INFO: Job input received: {input_data}")

    s3_file_path = input_data.get("s3_file_path")

    if not s3_file_path:
        print("WARNING: No 's3_file_path' provided in RunPod job input.", file=sys.stderr)
        # Return an error compatible with RunPod's expected output format
        return {"error": "No 's3_file_path' provided in job input."}

    print(f"INFO: RunPod handler processing s3_file_path: {s3_file_path}")
    try:
        result = process_audio_for_bpm(s3_file_path)
        print(f"INFO: RunPod handler returning successful result: {result}")
        return result # RunPod expects a dictionary as a successful response
    except Exception as e:
        print(f"ERROR: RunPod handler failed to analyze BPM: {e}", file=sys.stderr)
        # Return an error compatible with RunPod's expected output format
        return {"error": f"Failed to analyze BPM: {str(e)}"}


# --- Main Execution Block ---
# This decides whether to run as a local Flask app or a RunPod serverless worker.
if __name__ == '__main__':
    # Check if we are running in a RunPod serverless environment
    # RunPod sets this environment variable
    if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
        print("INFO: Detected RUNPOD_SERVERLESS_MODE = true. Starting RunPod Serverless worker...")
        runpod.serverless.start({"handler": handler})
    else:
        # For local testing with Flask
        print("INFO: Not in RUNPOD_SERVERLESS_MODE. Starting Flask app for local development...")
        # IMPORTANT: For local testing, ensure your AWS credentials and S3_BUCKET_NAME
        # are set as environment variables in your local terminal.
        # Example (Linux/macOS):
        # export S3_BUCKET_NAME="your-test-bucket"
        # export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY"
        # export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_KEY"
        # python your_app_file.py
        app.run(host='0.0.0.0', port=5050, debug=True)
