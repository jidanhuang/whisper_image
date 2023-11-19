import whisper
import os
os.system('ffmpeg -i testdata/6375c17e61428d75.ogg 6375c17e61428d75.wav')
model = whisper.load_model("base")
result = model.transcribe("6375c17e61428d75.wav")
print(result["text"])
