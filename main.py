import os
import subprocess
import whisper
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Set up OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# dowload audio from youtube video using yt-dlp
def dowload_audio(youtube_url, output_path="audio.mp3"):
    command = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--output", output_path,
        youtube_url
    ]
    subprocess.run(command, check=True)
    print(f"Audio downloaded to {output_path}")

# transcribe audio using openai whisper tool
def transcribe_audio(audio_path):
    model = whisper.load_model("large-v3")
    result = model.transcribe(audio_path, language="it")
    print("Transcription complete!")
    return result["text"]

# summarize text using openai gpt-4
def summarize_text(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text for 90% of the original length and make test formal and concise."},
            {"role": "user", "content": f"Summarize the following text: {text}"}
        ]
    )
    summary = response.choices[0].message.content
    print("Summary complete!")
    return summary

# Main workflow
def process_youtube_video(youtube_url):
    audio_path = "audio.mp3"
    #dowload_audio(youtube_url, audio_path)
    transcription = transcribe_audio(audio_path)
    summary = summarize_text(transcription)

    print("\n--- Transcribing ---\n")
    print(transcription[:1000] + '...')
    print("\n--- Summary ---\n")

    # save 2 file: transcription.txt and summary.txt
    with open('transcription.txt', 'w', encoding='utf-8') as f:
        f.write(transcription)

    with open('summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)

if __name__ == "__main__":
    video_url = input("Enter the YouTube video URL: ")
    process_youtube_video(video_url)
    print("Done!")
