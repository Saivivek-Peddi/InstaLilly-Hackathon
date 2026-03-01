#!/usr/bin/env python3
"""
Test Voice Assistant with Audio Input using Gemma 3n

1. Generate test audio files using gTTS
2. Transcribe audio using Gemma 3n
3. Call functions based on transcription
"""

import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

# Generate test audio files
def create_test_audio():
    """Create test audio files using gTTS."""
    from gtts import gTTS
    from pydub import AudioSegment

    test_commands = [
        ("play_jazz.wav", "Play some jazz music"),
        ("navigate_airport.wav", "Navigate to the airport"),
        ("call_mom.wav", "Call mom"),
    ]

    audio_dir = Path("models/voice_assistant/test_audio")
    audio_dir.mkdir(exist_ok=True)

    for filename, text in test_commands:
        filepath = audio_dir / filename
        if not filepath.exists():
            print(f"Creating: {filename}")
            tts = gTTS(text=text, lang='en')
            # Save as mp3 first
            mp3_path = filepath.with_suffix('.mp3')
            tts.save(str(mp3_path))

            # Convert mp3 to wav using pydub
            audio = AudioSegment.from_mp3(str(mp3_path))
            # Convert to mono 16kHz
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(str(filepath), format="wav")
            os.remove(mp3_path)
        else:
            print(f"Exists: {filename}")

    return audio_dir


def load_gemma3n():
    """Load Gemma 3n model for audio processing."""
    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

    model_name = "google/gemma-3n-e2b-it"
    print(f"Loading {model_name}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()
    print("Model loaded!")

    return model, processor


def transcribe_audio(model, processor, audio_path: str) -> str:
    """Transcribe audio file using Gemma 3n."""
    # Load audio using soundfile
    audio_array, sr = sf.read(audio_path, dtype='float32')

    # Ensure mono
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        import scipy.signal as signal
        num_samples = int(len(audio_array) * 16000 / sr)
        audio_array = signal.resample(audio_array, num_samples)
        sr = 16000

    # Format as chat messages with audio (proper Gemma 3n format)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": "Transcribe this audio exactly."}
            ]
        }
    ]

    # Process with apply_chat_template
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

    # Decode response
    response = processor.decode(outputs[0], skip_special_tokens=True)
    # Extract transcription (after model tag)
    if "model" in response:
        transcription = response.split("model")[-1].strip()
    else:
        transcription = response.split("exactly.")[-1].strip()

    return transcription


def classify_command(transcription: str) -> dict:
    """Classify transcription into function call."""
    text = transcription.lower()

    if "play" in text and ("music" in text or "jazz" in text or "song" in text):
        genre = "jazz" if "jazz" in text else "unknown"
        return {"function": "play_music", "params": {"genre": genre}}
    elif "navigate" in text or "directions" in text or "go to" in text:
        return {"function": "navigate", "params": {"destination": "airport"}}
    elif "call" in text:
        contact = "mom" if "mom" in text else "unknown"
        return {"function": "make_call", "params": {"contact": contact}}
    elif "weather" in text:
        return {"function": "get_weather", "params": {}}
    elif "ac" in text or "temperature" in text or "climate" in text:
        return {"function": "climate_control", "params": {}}
    else:
        return {"function": "unknown", "params": {}}


def main():
    print("=" * 60)
    print("Voice Assistant - Audio Input Test (Gemma 3n E2B-IT)")
    print("=" * 60)

    # Create test audio files
    print("\n[1] Creating test audio files...")
    audio_dir = create_test_audio()

    # Load Gemma 3n
    print("\n[2] Loading Gemma 3n...")
    model, processor = load_gemma3n()

    # Test each audio file
    print("\n[3] Testing audio transcription and function calling...")
    print("-" * 60)

    test_files = [
        ("play_jazz.wav", "play_music"),
        ("navigate_airport.wav", "navigate"),
        ("call_mom.wav", "make_call"),
    ]

    for filename, expected_func in test_files:
        audio_path = audio_dir / filename
        print(f"\nAudio: {filename}")

        # Transcribe
        transcription = transcribe_audio(model, processor, str(audio_path))
        print(f"Transcription: \"{transcription}\"")

        # Classify
        func_call = classify_command(transcription)
        status = "OK" if func_call["function"] == expected_func else "X "
        print(f"Function: [{status}] {func_call}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
