import logging
import os
import random
from dotenv import load_dotenv

from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero, murf

load_dotenv()
logger = logging.getLogger("improv-host")
logger.setLevel(logging.INFO)

# --- SCENARIOS ---
SCENARIOS = [
    "You are a barista telling a customer their latte is a portal to another dimension.",
    "You are a time-traveler explaining TikTok to a medieval peasant.",
    "You are a cat trying to explain quantum physics to a dog.",
    "You are a superhero whose only power is turning things into soup.",
    "You are a tour guide at a museum for 'Haunted Furniture'."
]

# --- THE AGENT ---
class ImprovHost(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are the energetic host of 'Improv Battle'.
            
            ROLE:
            1. You set a scene for the player.
            2. You listen to their performance.
            3. You react (laugh, critique, or applaud).
            4. You move to the next round.
            
            TONE: Witty, loud, game-show style.
            """,
            
            # HEARING: OpenAI Whisper (Robust for your network)
            stt=openai.STT(),
            
            # BRAIN: OpenAI GPT-4o-mini
            llm=openai.LLM(model="gpt-4o-mini"),
            
            # VOICE: Murf Falcon (Challenge Requirement)
            tts=murf.TTS(
                model="FALCON",
                api_key=os.getenv("MURF_API_KEY")
            ),
            
            vad=silero.VAD.load(min_speech_duration=0.1)
        )
        self.round = 0

    async def on_enter(self) -> None:
        await self.session.say("Ladies and gentlemen! Welcome to the Grand Finale... IMPROV BATTLE! Are you ready to perform?")

    # Override text handler to drive the game via chat
    async def on_user_speech(self, text: str):
        # If they say yes/ready/start, start Round 1
        if self.round == 0:
            self.round = 1
            scene = random.choice(SCENARIOS)
            await self.session.say(f"Fantastic! Round 1. Here is your scenario: {scene}. ... ACTION!")
            return

        # If they are performing (Round > 0), the LLM handles the critique naturally.
        # We just log it here.
        print(f"User performed: {text}")

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=True)
    
    # 1. Create the Agent
    agent = ImprovHost()
    
    # 2. Create the Session (Empty init to prevent TypeError)
    session = AgentSession()
    
    # 3. Start the session (Passing the agent here works on all versions)
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))