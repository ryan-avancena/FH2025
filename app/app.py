from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from face_landmarks import generate_face_mesh_stream
import os
from werkzeug.utils import secure_filename
import os, threading, tempfile, wave

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# import pyaudio
# import whisper
# from whisper_demo import transcribe_audio
# model = whisper.load_model("small")
# latest_transcript = ""

# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# CHUNK = 1024
# RECORD_SECONDS = 5

# def mic_transcription_loop():
#     global latest_transcript
#     audio = pyaudio.PyAudio()

#     stream = audio.open(format=FORMAT,
#                         channels=CHANNELS,
#                         rate=RATE,
#                         input=True,
#                         frames_per_buffer=CHUNK)

#     print("🎙️ Mic transcription started")
#     try:
#         while True:
#             frames = []
#             for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#                 data = stream.read(CHUNK, exception_on_overflow=False)
#                 frames.append(data)

#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
#                 temp_filename = temp_audio.name

#             with wave.open(temp_filename, 'wb') as wf:
#                 wf.setnchannels(CHANNELS)
#                 wf.setsampwidth(audio.get_sample_size(FORMAT))
#                 wf.setframerate(RATE)
#                 wf.writeframes(b''.join(frames))

#             result = model.transcribe(temp_filename)
#             latest_transcript = result["text"]
#             print("📝", latest_transcript)

#     except Exception as e:
#         print("⚠️", e)
#     finally:
#         stream.stop_stream()
#         stream.close()
#         audio.terminate()


# threading.Thread(target=mic_transcription_loop, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_face_mesh_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/transcribe', methods=['POST'])
# def transcribe():
#     if 'audio' not in request.files:
#         return "No audio file uploaded", 400

#     audio = request.files['audio']
#     filename = secure_filename(audio.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     audio.save(filepath)

#     transcript = transcribe_audio(filepath)
#     return render_template('index.html', transcript=transcript)

# @app.route('/live_transcript')
# def live_transcript():
#     return jsonify(text=latest_transcript)

if __name__ == "__main__":
    app.run(debug=True)
