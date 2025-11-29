import logging
import os
import json
import random
from typing import Annotated, List
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
# Import OpenAI and Murf
from livekit.plugins import deepgram, silero, openai, murf

load_dotenv()
logger = logging.getLogger("game-master")
logger.setLevel(logging.INFO)

# --- GAME STATE ---
@dataclass
class Character:
    name: str = "Hero"
    hp: int = 100
    inventory: List[str] = None
    def __post_init__(self):
        if self.inventory is None: self.inventory = ["Sword", "Potion"]

@dataclass
class GameState:
    location: str = "The Dark Forest"
    turn: int = 1
    character: Character = None
    def __post_init__(self):
        if self.character is None: self.character = Character()
    def to_json(self):
        return json.dumps(asdict(self))

RunContext_T = RunContext[GameState]

# --- THE INTELLIGENT AGENT ---
class GameMasterAgent(Agent):
    def __init__(self):
        self.game_state = GameState()
        
        super().__init__(
            instructions=f"""You are the Dungeon Master for a D&D fantasy adventure.
            
            CURRENT STATE: {self.game_state.to_json()}
            
            RULES:
            1. Describe the scene vividly but briefly (2 sentences max).
            2. Offer the player a choice.
            3. Use 'roll_dice' if the player fights, climbs, or takes risks.
            4. Use 'manage_inventory' if they pick up items.
            
            Make it exciting!""",
            
            # 1. HEARING: OpenAI STT (Network Friendly)
            stt=openai.STT(),
            
            # 2. BRAIN: OpenAI GPT-4o (Smart & Creative)
            llm=openai.LLM(model="gpt-4o-mini"),
            
            # 3. VOICE: Murf Falcon (Challenge Requirement)
            tts=murf.TTS(
                model="FALCON",
                api_key=os.getenv("MURF_API_KEY")
            ),
            vad=silero.VAD.load(min_speech_duration=0.1)
        )

    async def on_enter(self) -> None:
        await self.session.say("Welcome, adventurer. The forest looms before you. Shadows dance between the trees. What do you do?")

    # --- TOOLS ---
    @function_tool
    async def roll_dice(self, context: RunContext_T, action: Annotated[str, "Action description"], difficulty: int = 10) -> str:
        """Roll a D20 for risky actions."""
        roll = random.randint(1, 20)
        outcome = "SUCCESS" if roll >= difficulty else "FAILURE"
        return f"Rolled a {roll}. {outcome}!"

    @function_tool
    async def manage_inventory(self, context: RunContext_T, action: Annotated[str, "'add' or 'remove'"], item: str) -> str:
        """Add or remove items."""
        state = self.game_state
        if action == "add":
            state.character.inventory.append(item)
            return f"Added {item}."
        elif action == "remove":
            if item in state.character.inventory:
                state.character.inventory.remove(item)
                return f"Removed {item}."
        return "Done."

# --- ENTRYPOINT ---
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=True)
    agent = GameMasterAgent()
    session = AgentSession()
    session.userdata = agent.game_state
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))