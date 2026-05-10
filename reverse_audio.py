from pydub import AudioSegment

# Load the most recent step file
audio = AudioSegment.from_mp3("step_008800.mp3")

# Reverse the audio
reversed_audio = audio.reverse()

# Export as mp3
reversed_audio.export("reversed.mp3", format="mp3")

print(f"Done! Original duration: {len(audio)}ms, Reversed duration: {len(reversed_audio)}ms")
print("Saved to reversed.mp3")
