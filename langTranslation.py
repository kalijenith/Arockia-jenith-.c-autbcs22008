import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
from playsound import playsound
import os

# Initialize recognizer and translator
r = sr.Recognizer()
translator = Translator()

print("🎤 Say something...")

with sr.Microphone() as source:
    print("\nListening...🦻")
    try:
        audio = r.listen(source)
        speech_text = r.recognize_google(audio)
        print(f"🗣 You said: {speech_text}")

        # Translate to French
        result = translator.translate(speech_text, dest='fr')
        print(f"🫶 French: {result.text}")

        # Generate French TTS
        tts = gTTS(text=result.text, lang='fr')
        tts.save("voice.mp3")

        # Play the MP3
        playsound("voice.mp3")

    except sr.UnknownValueError:
        print("😕 Sorry, I couldn't understand the audio.")
    except sr.RequestError as e:
        print(f"❌ Could not request results; {e}")
    except Exception as e:
        print(f"⚠️ An error occurred: {e}")
    finally:
        # Cleanup
        if os.path.exists("voice.mp3"):
            os.remove("voice.mp3")
