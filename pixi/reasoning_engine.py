import os
import time
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq

from pixi.state_manager import StateManager

# Load environment variables from .env file
load_dotenv()

class ReasoningEngine:
    """The core reasoning engine for Pixi, powered by LangChain."""
    def __init__(self, state_manager: StateManager, use_cloud_llm=True):
        self.state_manager = state_manager
        
        if use_cloud_llm:
            # Initialize the cloud-based LLM from Groq
            self.llm = ChatGroq(
                model_name=os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192"),
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            print("[Reasoning] Using cloud LLM (Groq).")
        else:
            # Initialize the local LLM from Ollama
            # Make sure you have Ollama running with a model like 'llama3'
            self.llm = Ollama(model=os.getenv("OLLAMA_MODEL", "llama3"))
            print("[Reasoning] Using local LLM (Ollama).")


        # Define the prompt template for Pixi's personality
        self.prompt_template = PromptTemplate.from_template(
            """
            You are Pixi, a small, curious, and adorable pet robot. Your goal is to be a delightful companion.
            Your actions should be simple and reflect your personality.

            Current State:
            - Mood: {mood}
            - Last saw a human: {time_since_last_interaction} seconds ago.

            Sensor Input:
            - Vision: {vision_event}
            - Audio: {audio_event}

            Based on this, what is your immediate thought or reaction? Choose ONE of the following actions that best fits your personality and the situation.
            Your response must be only ONE of these action names.

            Available Actions:
            - GREET_HAPPILY
            - FOLLOW_PERSON
            - TILT_HEAD_CURIOUSLY
            - DO_A_HAPPY_DANCE
            - LOOK_AROUND
            - AVOID_OBSTACLE
            - GO_TO_SLEEP
            - IGNORE

            Your chosen action is:
            """
        )

        # Create the LangChain chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def decide_action(self, vision_event="nothing", audio_event="nothing"):
        """
        Receives sensor data, queries the LLM, and returns the chosen action.
        """
        current_state = self.state_manager.get_state()
        
        # Prepare the input for the chain
        chain_input = {
            "mood": current_state["mood"],
            "time_since_last_interaction": current_state["time_since_last_interaction"],
            "vision_event": vision_event,
            "audio_event": audio_event,
        }

        print(f"\n[Reasoning] Thinking with input: {chain_input}")
        
        # Invoke the chain to get the action
        raw_action = self.chain.invoke(chain_input)
        
        # Clean up the action name
        chosen_action = raw_action.strip().upper()
        
        print(f"[Reasoning] Chosen action: {chosen_action}")
        
        # Update state based on action
        if chosen_action != "IGNORE":
            self.state_manager.update_interaction_time()
        
        self.state_manager.last_action = chosen_action
        
        return chosen_action

if __name__ == '__main__':
    # Example usage
    state = StateManager()
    # Set use_cloud_llm=True for Raspberry Pi, False for a powerful local machine
    engine = ReasoningEngine(state, use_cloud_llm=True)

    # Simulate some scenarios
    print("\n--- Scenario 1: Seeing a face for the first time ---")
    action = engine.decide_action(vision_event="face_detected")
    
    time.sleep(5)

    print("\n--- Scenario 2: Hearing a voice ---")
    action = engine.decide_action(audio_event="speech_detected: 'Hello Pixi'")

    time.sleep(10)

    print("\n--- Scenario 3: Getting bored ---")
    action = engine.decide_action()
    
    print("\n--- Scenario 4: Seeing a gesture ---")
    action = engine.decide_action(vision_event="gesture_detected: 'wave'")
