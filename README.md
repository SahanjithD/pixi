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
│   ├── reasoning_engine.py  # LangChain chain + Groq integration
│   └── state_manager.py     # Robot state tracking
├── requirements.txt         # Python dependencies
├── instructions.md          # Detailed architecture notes
└── README.md
```

## Prerequisites

- Python 3.10+
- Groq API key ([request one](https://console.groq.com/keys))
- (Optional) Local LLM runtime such as Ollama for development on powerful hardware

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
   ```

## Running the Reasoning Engine

Simulate Pixi's decision making loop:
```powershell
python -m pixi.reasoning_engine
```
The script prints sensor events, the selected action, and a short explanation of Pixi's behavior.

## Next Steps

- **Perception Integration:** Add camera and microphone pipelines to feed real sensor data into the reasoning engine.
- **Action Layer:** Connect `GREET_HAPPILY`, `FOLLOW_PERSON`, and other actions to the robot's actuators and expressive outputs.
- **Behavior Tuning:** Adjust the prompt, moods, and state transitions to refine Pixi's personality.

## License

MIT
