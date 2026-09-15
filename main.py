import whisper


# ==========================================
# SIH26172 - Voice Activator Prototype
# ==========================================

AUDIO_FILE = "audio/test.wav"

# Change this to your actual wake word
WAKE_WORD = "hey nova"


def transcribe_audio(audio_file):
    print("\nLoading Whisper model...")

    model = whisper.load_model("base")

    print("Transcribing audio...\n")

    result = model.transcribe(
        audio_file,
        fp16=False
    )

    return result["text"].strip()


def check_wake_word(transcription, wake_word):

    transcription = transcription.lower()
    wake_word = wake_word.lower()

    if wake_word in transcription:
        return True

    return False


def main():

    print("=" * 50)
    print(" SIH26172 - Voice Activator")
    print("=" * 50)

    # Step 1: Transcribe
    transcription = transcribe_audio(AUDIO_FILE)

    print("Transcription:")
    print(transcription)

    # Step 2: Check wake word
    detected = check_wake_word(
        transcription,
        WAKE_WORD
    )

    print("\n" + "-" * 50)

    if detected:
        print("WAKE WORD: DETECTED ✓")
        print("Voice activator triggered!")

    else:
        print("WAKE WORD: NOT DETECTED ✗")
        print("Voice activator remains inactive.")

    print("-" * 50)


if __name__ == "__main__":
    main()