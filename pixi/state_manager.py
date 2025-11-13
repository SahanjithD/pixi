import time

class StateManager:
    """Manages the robot's internal state."""
    def __init__(self):
        self.mood = "curious"  # Default mood
        self.last_interaction_time = time.time()
        self.recognized_person = None
        self.last_action = None

    def update_mood(self, new_mood):
        """Updates the robot's mood."""
        if self.mood != new_mood:
            print(f"[State] Mood changed from {self.mood} to {new_mood}")
            self.mood = new_mood

    def update_interaction_time(self):
        """Resets the interaction timer."""
        self.last_interaction_time = time.time()

    def time_since_last_interaction(self):
        """Returns the time in seconds since the last interaction."""
        return time.time() - self.last_interaction_time

    def get_state(self):
        """Returns a dictionary of the current state."""
        return {
            "mood": self.mood,
            "time_since_last_interaction": round(self.time_since_last_interaction(), 1),
            "recognized_person": self.recognized_person,
        }
