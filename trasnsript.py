from octoai.client import Client
import base64
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OCTOAI_API_TOKEN"] = os.getenv("OCTOAI_API_TOKEN")

whisper_url = "https://whisper-demo-kk0powt97tmb.octoai.run/predict"
whisper_health_check = "https://whisper-demo-kk0powt97tmb.octoai.run/healthcheck"

# First, we need to convert an audio file to base64.
file_path = "Math_isn't_hard_it's_a_language _Randy_Palisoc _TEDxManhattanBeach.mp3"
with open(file_path, "rb") as f:
    encoded_audio = base64.b64encode(f.read())
    base64_string = encoded_audio.decode("utf-8")

# These are the inputs we will send to the endpoint, including the audio base64 string.
inputs = {
    "language": "en",
    "task": "transcribe",
    "audio": base64_string,
}

OCTOAI_TOKEN = "OCTOAI_API_TOKEN"
# The client will also identify if OCTOAI_TOKEN is set as an environment variable
# So if you have it set, you can simply use:
# client = Client()
client = Client(token=OCTOAI_TOKEN)
if client.health_check(whisper_health_check) == 200:
  outputs = client.infer(endpoint_url=whisper_url, inputs=inputs)
  transcription = outputs["transcription"]
  assert "She sells seashells by the seashore" in transcription
  assert (
        "She sells seashells by the seashore"
        in outputs["response"]["segments"][0]["text"]
    )