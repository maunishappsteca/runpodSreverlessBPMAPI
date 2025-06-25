import os
from dotenv import load_dotenv # Add this line

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
if S3_BUCKET: # Only initialize if bucket name is provided
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
else:
    print("WARNING: S3_BUCKET_NAME environment variable not set. S3 functionality will not work.")
    s3_client = None # Set to None if S3 is not configured

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the core logic of your application as a standalone function
# This makes it reusable by both Flask (for local testing) and RunPod's handler
def process_audio_for_bpm(s3_file_path):
    if not S3_BUCKET or not s3_client:
        raise ValueError("S3 bucket or client not configured.")

    filename = os.path.basename(s3_file_path)
    local_file_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        # Download the file from S3
        print(f"Downloading {s3_file_path} from S3 bucket {S3_BUCKET} to {local_file_path}")
        s3_client.download_file(S3_BUCKET, s3_file_path, local_file_path)
        print("Download complete.")

        # Load audio and estimate BPM
        y, sr = librosa.load(local_file_path, sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        return {"bpm": round(float(tempo), 2)}

    except Exception as e:
        import traceback
        traceback.print_exc() # Print full traceback for debugging in logs
        raise e # Re-raise the exception to be caught by the caller/handler
    finally:
        # Clean up
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
            print(f"Cleaned up temporary file: {local_file_path}")


# --- Flask Route for Local Development/Testing (Optional) ---
# This allows you to test your Flask app locally before deploying to RunPod.
# For production deployment on RunPod, this route won't be directly hit,
# but the 'handler' function below will use the 'process_audio_for_bpm' logic.
@app.route("/analyze-bpm", methods=["POST"])
def analyze_bpm_flask_route():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    if "s3_file_path" not in request.json:
        return jsonify({"error": "No 's3_file_path' provided in JSON"}), 400

    s3_file_path = request.json["s3_file_path"]

    try:
        result = process_audio_for_bpm(s3_file_path)
        return jsonify(result)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 500
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# --- RunPod Serverless Handler (REQUIRED for Deployment) ---
# This function is the entry point for your RunPod Serverless worker.
# It receives a 'job' dictionary containing the input data.
def handler(job):
    # The input to your API will be in job['input']
    input_data = job.get('input', {})

    s3_file_path = input_data.get("s3_file_path")

    if not s3_file_path:
        # Return an error compatible with RunPod's expected output format
        return {"error": "No 's3_file_path' provided in job input."}

    try:
        result = process_audio_for_bpm(s3_file_path)
        return result # RunPod expects a dictionary as a successful response
    except Exception as e:
        # Return an error compatible with RunPod's expected output format
        return {"error": f"Failed to analyze BPM: {str(e)}"}


# --- Main Execution Block ---
# This decides whether to run as a local Flask app or a RunPod serverless worker.
if __name__ == '__main__':
    # Check if we are running in a RunPod serverless environment
    # RunPod sets this environment variable
    if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
        print("Starting RunPod Serverless worker...")
        runpod.serverless.start({"handler": handler})
    else:
        # For local testing with Flask
        print("Starting Flask app for local development...")
        # IMPORTANT: For local testing, ensure your AWS credentials and S3_BUCKET_NAME
        # are set as environment variables in your local terminal.
        # Example (Linux/macOS):
        # export S3_BUCKET_NAME="your-test-bucket"
        # export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY"
        # export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_KEY"
        # python your_app_file.py
        app.run(host='0.0.0.0', port=5050, debug=True)
