# 🐾 AI System for Pet Robot using LangChain

## 🧠 Overview

This document explains how to implement the **AI module** of a **pet robot** (similar to Emo or Vector) using **LangChain**.  
The system allows the robot to **perceive, understand, and interact** with humans intelligently using **vision, audio, and contextual reasoning**.

---

## ⚙️ System Architecture 

The robot's intelligence is structured in four main layers. This modular design allows for clear separation of concerns, making the system easier to build, debug, and upgrade.

```mermaid
graph TD
    A[Sensors] --> B(Data Processing Layer);
    B --> C(LangChain Reasoning Layer);
    C --> D[Action Layer];

    subgraph "Perception"
        A(Camera & Microphone)
    end

    subgraph "Data Processing"
        B
        B1[Vision: OpenCV & MediaPipe<br>Face/Gesture Recognition]
        B2[Audio: Speech Recognition<br>Speech-to-Text]
        A --> B1;
        A --> B2;
    end

    subgraph "Cognition"
        C
        C1[State Manager<br>Mood, Memory]
        C2[LLM Chain (LangChain)<br>Decision Making]
        B1 --> C1;
        B2 --> C1;
        C1 --> C2;
    end

    subgraph "Execution"
        D
        D1[Robot Controller<br>Motors, Screen, Speaker]
        C2 --> D1;
    end
```

### 1. Sensors
This is the robot's interface to the physical world.
- **Camera:** Captures the video stream for visual processing.
- **Microphone:** Records audio for speech recognition and sound analysis.
- **(Optional) Proximity/IR Sensors:** For real-time obstacle avoidance, which should operate at a lower level than the AI reasoning for safety.

### 2. Data Processing Layer
This layer transforms raw sensor data into a structured format that the reasoning layer can understand.
- **Vision Processing (OpenCV & MediaPipe):**
    - **Capture:** Use `OpenCV` to read frames from the camera.
    - **Detect:** Use `MediaPipe`'s pre-trained models for:
        - **Face Detection:** To know when a human is present and get their location in the frame.
        - **Facial Expression Recognition:** Analyze facial landmarks to infer emotions (e.g., smiling, surprised).
        - **Gesture Recognition:** Identify specific hand gestures (e.g., a wave, a thumbs-up).
    - **Output:** Generate simple text descriptions like `{"event": "face_detected", "position": [x, y], "emotion": "smile"}`.
- **Audio Processing (SpeechRecognition):**
    - **Transcribe:** Use a library like `SpeechRecognition` (potentially with an engine like Whisper) to convert spoken words into text.
    - **Output:** `{"event": "speech_detected", "text": "hello Pixi"}`.

### 3. LangChain Reasoning Layer (The "Brain")
This is the core of the robot's personality and decision-making. It receives the structured data from the processing layer and decides what to do next.
- **State Manager:** A simple class that holds the robot's current state, such as:
    - `mood`: (e.g., "happy", "curious", "sleepy")
    - `last_interaction_time`: To determine if the robot should get "bored" and act on its own.
    - `recognized_person`: The name of the person it's interacting with.
- **LLM Chain (LangChain):**
    - **Prompt Template:** This is where you define Pixi's personality. The prompt will be fed with the current state and sensor data.
      ```
      You are Pixi, a small, curious, and adorable pet robot. Your goal is to be a delightful companion.

      Current State:
      - Mood: {mood}
      - Last saw a human: {time_since_last_seen} seconds ago.

      Sensor Input:
      - Vision: {vision_event}
      - Audio: {audio_event}

      Based on this, what is your immediate thought or reaction? Choose ONE of the following actions that best fits your personality and the situation:
      - GREET_HAPPILY
      - FOLLOW_PERSON
      - TILT_HEAD_CURIOUSLY
      - DO_A_HAPPY_DANCE
      - LOOK_AROUND
      - AVOID_OBSTACLE
      - GO_TO_SLEEP
      - IGNORE

      Your chosen action is:
      ```
    - **LLM:** A fast and efficient language model (e.g., a local Ollama model like Llama 3, or a fast API-based one).
    - **Output Parser:** A LangChain `StrOutputParser` to extract the chosen action (e.g., `GREET_HAPPILY`).

### 4. Action Layer
This layer takes the abstract decision from the reasoning layer and translates it into physical commands for the robot's hardware.
- **Action Mapping:** A dictionary or switch-case structure that maps action names to functions.
  ```python
  action_map = {
      "GREET_HAPPILY": robot.greet,
      "FOLLOW_PERSON": robot.start_following,
      "TILT_HEAD_CURIOUSLY": robot.tilt_head,
      # ... and so on
  }
  ```
- **Robot Controller:** A set of functions that directly control the hardware.
    - `robot.motors.move(x, y)`
    - `robot.screen.display_emotion('happy_eyes')`
    - `robot.speaker.play_sound('greeting.wav')`
    - `robot.stop_all()`

##  Behavioral Patterns Implementation

- **Random Playful Actions:** In your main loop, if the `State Manager` reports that it's been a while since the last interaction, you can periodically trigger the reasoning layer with a "boredom" event. The LLM, guided by its personality, might then choose an action like `LOOK_AROUND` or `DO_A_HAPPY_DANCE`.
- **Following Humans:** When the vision system reports a `face_detected` event with coordinates, the reasoning layer can decide to `FOLLOW_PERSON`. The action layer then calls a function that uses these coordinates as a target for the motor control loop.
- **Avoiding Obstacles:** This should be a high-priority, low-level process. The reasoning layer can be notified of an obstacle, but the actual stopping or turning should be handled almost instantly by the motor controller based on proximity sensor data to ensure the robot doesn't run into things.
- **Curiosity and Adorableness:** This is achieved through a combination of:
    1.  **Prompt Engineering:** The core personality defined in the prompt.
    2.  **Action Design:** The actions you create (e.g., `TILT_HEAD_CURIOUSLY`) and the corresponding animations/sounds should be designed to be cute.
    3.  **State-Driven Behavior:** The robot's mood should influence its actions. A "happy" Pixi might react more energetically than a "sleepy" one.




