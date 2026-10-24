import os
from dotenv import load_dotenv
from gnani.tts import GnaniTTSClient

load_dotenv()

# Initializes using GNANI_API_KEY from environment
tts = GnaniTTSClient()

try:
    print("Synthesizing test speech with Gnani...")
    audio = tts.synthesize(
        text="Welcome to ZentroMart support.",
        voice="Kaveri",
        model="timbre-v2.5",
        language="en-IN"
    )
    with open("test_output.wav", "wb") as f:
        f.write(audio)
    print(" Success! Audio saved as test_output.wav")
except Exception as e:
    print(f" Error communicating with Gnani API: {e}")