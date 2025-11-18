# Pixi: LangChain-Powered Pet Robot

Pixi is a curious and adorable pet robot that perceives the world through vision and audio, reasons about its observations with LangChain, and responds using expressive behaviors. This repository contains the AI module that drives Pixi's personality and decision making.

## System Overview

```
[Sensors] → [Data Processing Layer] → [LangChain Reasoning Layer] → [Action Layer]
```

- **Sensors:** Cameras, microphones, and optional proximity sensors gather input from the environment.
- **Data Processing:** OpenCV and MediaPipe detect faces, expressions, and gestures while speech recognition converts audio to text events.
- **LangChain Reasoning:** A state manager tracks Pixi's mood and context. An LLM (via Groq) chooses the next action based on sensor events and internal state.
- **Action Layer:** Robot controllers translate the selected action into motor commands, display expressions, and sounds.

## Repository Structure

```
.
├── pixi/
│   ├── core/                # State management, actions, reasoning
│   │   ├── actions.py
│   │   ├── reasoning_engine.py
│   │   └── state_manager.py
│   ├── audio/               # Hotword detection pipeline
│   │   └── hotword.py
│   ├── vision/              # MediaPipe-based perception & preview server
│   │   ├── perception.py
│   │   └── web_preview_server.py
│   └── runners/             # CLI entry points for different runtime modes
│       ├── run_audio_hotword.py
│       ├── run_realtime.py
│       └── run_vision_only.py
├── requirements.txt         # Python dependencies
├── instructions.md          # Detailed architecture notes
├── scripts/
│   └── web_preview.py       # Legacy CLI wrapper (uses new vision modules)
└── README.md
```

## Prerequisites

- Python 3.10+
- Groq API key ([request one](https://console.groq.com/keys))
- (Optional) Local LLM runtime such as Ollama for development on powerful hardware
- Picovoice AccessKey for Porcupine hotword detection ([create one](https://console.picovoice.ai/))

## Setup

1. Clone the repository (or copy the project) and create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Configure environment variables in `.env`:
   ```ini
   GROQ_API_KEY="your_api_key"
   GROQ_MODEL_NAME="llama-3.1-8b-instant"
   # Optional for local LLM fallback
   # OLLAMA_MODEL=llama3
   # Required for Porcupine hotword detection
    PICOVOICE_ACCESS_KEY="your_picovoice_access_key"
   ```

## Running the Reasoning Engine

Simulate Pixi's decision making loop:
```powershell
python -m pixi.core.reasoning_engine
```
The script prints sensor events, the selected action, and a short explanation of Pixi's behavior.

## Testing Vision Processing Only

Run the vision loop without the reasoning engine:

```powershell
python -m pixi.runners.run_vision_only --display
```

Remove `--display` if you only want console logs.

## Testing Audio Hotword Detection Only

Run Porcupine in isolation to validate your microphone and keywords:

```powershell
python -m pixi.runners.run_audio_hotword --hotword-file "Wake-up-buddy_en_raspberry-pi_v3_0_0/Wake-up-buddy_en_raspberry-pi_v3_0_0.ppn"
```

Omit `--hotword-file` to listen for the default `picovoice` wake word, or repeat `--hotword-keyword` for multiple built-in triggers. Use `pvrecorder --show_audio_devices` to discover input device indices and pass one with `--hotword-device-index` if needed.

## Browser Camera Preview

When you want a smoother camera view than the OpenCV window provides, launch the MJPEG preview server:

```powershell
python -m scripts.web_preview --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000/` in your browser to watch the stream with the same face/gesture overlays shown in the OpenCV preview. Use `--no-annotations` if you prefer raw frames. The camera device can only be owned by one process at a time, so stop the realtime loop (or run it with a prerecorded source) when you launch the preview server.

## Hotword Detection (Porcupine)

To iterate on audio-only interactions, enable the Picovoice Porcupine listener while disabling the camera pipeline:

```powershell
python -m pixi.runners.run_realtime --disable-vision --enable-hotword
```

By default Pixi listens for the built-in `picovoice` wake word. Supply additional built-in keywords with `--hotword-keyword porcupine` (can be repeated) or custom `.ppn` files via `--hotword-file path/to/keyword.ppn`. Select a specific microphone using `--hotword-device-index` (see `pvrecorder --show_audio_devices`).

## Next Steps

- **Perception Integration:** Add camera and microphone pipelines to feed real sensor data into the reasoning engine.
- **Action Layer:** Connect `GREET_HAPPILY`, `FOLLOW_PERSON`, and other actions to the robot's actuators and expressive outputs.
- **Behavior Tuning:** Adjust the prompt, moods, and state transitions to refine Pixi's personality.

## License

MIT
