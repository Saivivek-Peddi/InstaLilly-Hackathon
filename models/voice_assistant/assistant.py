#!/usr/bin/env python3
"""
Voice Assistant with Function Calling

Uses Gemma for:
1. Speech-to-Text (audio -> text) - future
2. Function calling (text -> action)
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Available functions for the vehicle
FUNCTIONS = {
    "play_music": {
        "description": "Play music in the car",
        "parameters": ["genre", "song", "artist"],
    },
    "stop_music": {
        "description": "Stop playing music",
        "parameters": [],
    },
    "navigate": {
        "description": "Navigate to a destination",
        "parameters": ["destination"],
    },
    "make_call": {
        "description": "Make a phone call",
        "parameters": ["contact"],
    },
    "send_message": {
        "description": "Send a text message",
        "parameters": ["contact", "message"],
    },
    "climate_control": {
        "description": "Control AC/heating",
        "parameters": ["action", "temperature"],
    },
    "window_control": {
        "description": "Control car windows",
        "parameters": ["action", "window"],
    },
    "get_weather": {
        "description": "Get weather information",
        "parameters": ["location"],
    },
    "set_reminder": {
        "description": "Set a reminder",
        "parameters": ["reminder", "time"],
    },
}


def build_prompt(user_command: str) -> str:
    """Build prompt for function calling using Gemma chat format."""
    lines = []
    for name, info in FUNCTIONS.items():
        lines.append(f"- {name}: {info['description']}")
    functions_desc = "\n".join(lines)

    prompt = f"""<start_of_turn>user
What function should be called for this voice command?
Command: "{user_command}"

Functions available:
{functions_desc}

Answer with JSON: {{"function": "name", "params": {{}}}}<end_of_turn>
<start_of_turn>model
"""
    return prompt


def parse_response(response: str) -> dict:
    """Parse model response to extract function call."""
    try:
        # Find JSON in response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
    except:
        pass
    return {"function": "unknown", "params": {}}


def execute_function(func_call: dict) -> str:
    """Simulate executing a function."""
    func = func_call.get("function", "unknown")
    params = func_call.get("params", {})

    if func == "play_music":
        genre = params.get("genre", "")
        song = params.get("song", "")
        artist = params.get("artist", "")
        if artist:
            return f"Playing songs by {artist}"
        elif song:
            return f"Playing {song}"
        elif genre:
            return f"Playing {genre} music"
        return "Playing music"

    elif func == "stop_music":
        return "Music stopped"

    elif func == "navigate":
        dest = params.get("destination", "unknown")
        return f"Navigating to {dest}"

    elif func == "make_call":
        contact = params.get("contact", "unknown")
        return f"Calling {contact}"

    elif func == "climate_control":
        action = params.get("action", "adjust")
        temp = params.get("temperature", "")
        if temp:
            return f"Setting temperature to {temp}"
        return f"Climate control: {action}"

    elif func == "get_weather":
        loc = params.get("location", "current location")
        return f"Getting weather for {loc}"

    elif func == "window_control":
        action = params.get("action", "toggle")
        window = params.get("window", "all")
        return f"Windows: {action} {window}"

    elif func == "send_message":
        contact = params.get("contact", "")
        return f"Sending message to {contact}"

    elif func == "set_reminder":
        reminder = params.get("reminder", "")
        return f"Reminder set: {reminder}"

    return "Sorry, I did not understand that command"


class VoiceAssistant:
    def __init__(self, model_name: str = "google/gemma-2-2b-it"):
        print(f"Loading model: {model_name}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.model.eval()
        print("Model loaded!")

    def process_command(self, command: str) -> dict:
        """Process a text command and return function call."""
        prompt = build_prompt(command)

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Get the model's response after the prompt
        response = response.split("model")[-1].strip() if "model" in response else response

        return parse_response(response)

    def handle_command(self, command: str) -> tuple:
        """Full pipeline: command -> function -> execution."""
        func_call = self.process_command(command)
        result = execute_function(func_call)
        return result, func_call


def main():
    print("=" * 60)
    print("Voice Assistant - Function Calling Test")
    print("=" * 60)

    assistant = VoiceAssistant()

    # Test commands
    test_commands = [
        "play some jazz music",
        "navigate to the airport",
        "call mom",
        "turn up the AC",
        "what is the weather today",
    ]

    print("\nTesting commands:")
    print("-" * 40)

    for cmd in test_commands:
        result, func_call = assistant.handle_command(cmd)
        print(f"\nCommand: \"{cmd}\"")
        print(f"Function: {func_call}")
        print(f"Result: {result}")


if __name__ == "__main__":
    main()
