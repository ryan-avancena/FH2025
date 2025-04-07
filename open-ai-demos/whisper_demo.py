import whisper
import pyaudio
import wave
import numpy as np
import tempfile

model = whisper.load_model("small")
result = model.transcribe("1d.mp3")
print(result["text"])


""" if you want to transcribe the audio from your microphone  ... """
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000  
# CHUNK = 2048
# RECORD_SECONDS = 5  

# audio = pyaudio.PyAudio()

# stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,
#                                   channels=1,
#                                   rate=16000,  # Ensure this matches your Whisper model's expected rate
#                                   input=True,
#                                   frames_per_buffer=2048)  # Larger buffer to prevent overflow

# print("listening ... ")

# try:
#     while True:
#         frames = []
#         for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#             data = stream.read(CHUNK, exception_on_overflow=False)
#             frames.append(data)

#         # Save the recorded chunk to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
#             temp_filename = temp_audio.name  # Store filename to use later

#         # Write to the temporary WAV file
#         with wave.open(temp_filename, 'wb') as wf:
#             wf.setnchannels(CHANNELS)
#             wf.setsampwidth(audio.get_sample_size(FORMAT))
#             wf.setframerate(RATE)
#             wf.writeframes(b''.join(frames))

#         # Transcribe using Whisper
#         result = model.transcribe(temp_filename)
#         print("You said:", result["text"])

# except KeyboardInterrupt:
#     print("\nStopping transcription.")
#     stream.stop_stream()
#     stream.close()
#     audio.terminate()