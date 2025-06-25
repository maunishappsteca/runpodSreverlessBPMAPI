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
        print("ERROR: S3 bucket name or client not configured when calling process_audio_for_bpm.", file=sys.stderr)
        raise ValueError("S3 bucket or client not configured.")

    filename = os.path.basename(s3_file_path)
    local_file_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        # Download the file from S3
        print(f"INFO: Attempting to download '{s3_file_path}' from S3 bucket '{S3_BUCKET}' to '{local_file_path}'...")
        s3_client.download_file(S3_BUCKET, s3_file_path, local_file_path)
        print(f"INFO: Download of '{filename}' complete.")

        # Load audio and estimate BPM
        print(f"INFO: Loading audio file '{local_file_path}' with librosa...")
        y, sr = librosa.load(local_file_path, sr=None, mono=True)
        print(f"INFO: Audio loaded. Sample rate: {sr}, duration: {len(y)/sr:.2f} seconds.")

        print("INFO: Estimating BPM using librosa.beat.beat_track...")
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        print(f"INFO: BPM estimation complete. Raw tempo: {tempo}.")

        return {"bpm": round(float(tempo), 2)}

    except Exception as e:
        print(f"ERROR: An error occurred during audio processing for '{s3_file_path}'.", file=sys.stderr)
        import traceback
        traceback.print_exc() # Print full traceback for debugging in logs
        raise e # Re-raise the exception to be caught by the caller/handler
    finally:
        # Clean up
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
            print(f"INFO: Cleaned up temporary file: {local_file_path}")
        else:
            print(f"INFO: Temporary file '{local_file_path}' not found for cleanup (might not have been created or already removed).")


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
