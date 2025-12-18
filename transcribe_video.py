import whisper

model = whisper.load_model("medium")   # 可选 tiny/base/small/medium/large
result = model.transcribe("/Users/zcyj/[UeY3zoggG08].mp4", verbose=True)


with open("transcription.txt", "w") as f:
    f.write(result["text"])

print(result["text"])
