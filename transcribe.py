from octoai.client import Client
import base64
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OCTOAI_API_TOKEN"] = os.getenv("OCTOAI_API_TOKEN")

whisper_url = "https://whisper-demo-kk0powt97tmb.octoai.run/predict"
whisper_health_check = "https://whisper-demo-kk0powt97tmb.octoai.run/healthcheck"

client = Client(token=os.getenv("OCTOAI_API_TOKEN"))


# Transcribe videos
def transcribe_video(video_dir: str, output_dir: str):
    # loop through videofiles in the directory
    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)
        # open the video file and encode it to base64
        with open(video_path, "rb") as f:
            encoded_video = base64.b64encode(f.read())
            base64_string = encoded_video.decode("utf-8")
        # create the input dictionary
        inputs = {
            "language": "en",
            "task": "transcribe",
            "audio": base64_string,
        }
        # send the input dictionary to the endpoint
        if client.health_check(whisper_health_check) == 200:
            outputs = client.infer(endpoint_url=whisper_url, inputs=inputs)
            transcription = outputs["transcription"]
        # save the response to a file
        video_file = video_file.replace(".mp4", ".txt")
        output_path = os.path.join(output_dir, video_file)
        # create output direcotry if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, "w") as f:
            f.write(transcription)
        print(f"Transcribed {video_file} to {output_path}")


transcribe_video("./videos", "./data")
