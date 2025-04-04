from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from face_landmarks import generate_face_mesh_stream
import os
import numpy as np
from werkzeug.utils import secure_filename
import os, threading, tempfile, wave

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

import pyaudio
import whisper
from whisper_demo import transcribe_audio
model = whisper.load_model("small")
latest_transcript = ""

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 2

def mic_transcription_loop():
    global latest_transcript
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("🎙️ Mic transcription started")
    try:
        while True:
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

            # Combine audio and normalize
            audio_data = b''.join(frames)
            audio_np = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0

            # Transcribe directly from NumPy array
            result = model.transcribe(audio_np, fp16=False)
            latest_transcript = result["text"]

            print("📝", latest_transcript)

    except Exception as e:
        print("⚠️", e)
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


threading.Thread(target=mic_transcription_loop, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_face_mesh_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # if 'audio' not in request.files:
    #     return "No audio file uploaded", 400

    audio_file = request.files['audio']
    audio_bytes = audio_file.read()

    audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
    result = model.transcribe(audio_np, fp16=False)
    transcript = result['text']

    return render_template('index.html', transcript=transcript)


@app.route('/live_transcript')
def live_transcript():
    return jsonify(text=latest_transcript)

if __name__ == "__main__":
    app.run(debug=True)
