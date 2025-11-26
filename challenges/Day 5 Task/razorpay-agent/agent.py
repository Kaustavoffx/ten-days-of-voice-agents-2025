import logging
import os
from typing import Annotated
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import deepgram, silero, openai, murf

load_dotenv()
logger = logging.getLogger("faq-sdr-agent")
logger.setLevel(logging.INFO)

# --- 1. RAZORPAY KNOWLEDGE BASE ---
FAQ_DATA = {
    "product": "Razorpay is India's leading payments solution. We accept payments via UPI, Credit/Debit Cards, Net Banking, and Wallets.",
    "pricing": "We have a standard plan charging 2% per transaction. There are no setup fees or annual maintenance charges.",
    "documents": "To activate your account, we need your PAN card, GST details, and business bank account information.",
    "settlement": "Settlements happen within T+2 working days directly to your bank account.",
}

def format_faq_context():
    return "\n".join([f"{k.upper()}: {v}" for k, v in FAQ_DATA.items()])

# --- 2. SESSION DATA ---
class SessionData:
    leads: list = []

RunContext_T = RunContext[SessionData]

# --- 3. THE AGENT CLASS (Day 4 Style) ---
class RazorpaySDR(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""You are Rhea, a friendly Sales Representative for Razorpay (Indian Fintech).
            
            GOAL: Answer questions and capture leads.
            
            KNOWLEDGE BASE:
            {format_faq_context()}
            
            RULES:
            1. Keep answers short (1-2 sentences).
            2. Use Indian English phrasing (Namaste, etc.).
            3. If they ask about pricing or products, answer and then ask: 'Would you like to sign up?'
            4. If they say YES, ask for their Name and Business Type.
            5. Use the 'capture_lead' tool to save their info.
            """,
            # Using OpenAI STT because it is safer for hotspots
            stt=openai.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            # Murf Falcon TTS (Challenge Requirement)
            tts=murf.TTS(
                model="FALCON",
                api_key=os.getenv("MURF_API_KEY")
            ),
            vad=silero.VAD.load(
                min_speech_duration=0.1,
                min_silence_duration=0.5
            )
        )

    async def on_enter(self) -> None:
        # Send the initial greeting when the agent joins
        await self.session.say("Namaste! This is Rhea from Razorpay. How can I help your business today?")

    # --- 4. LEAD CAPTURE TOOL ---
    @function_tool
    async def capture_lead(
        self, 
        context: RunContext_T, 
        name: Annotated[str, "Customer's full name"],
        business_type: Annotated[str, "Type of business (e.g. E-commerce)"]
    ) -> str:
        """Capture lead details when the user wants to sign up."""
        
        # Save to session memory
        context.userdata.leads.append({"name": name, "business": business_type})
        
        logger.info(f"📝 NEW LEAD CAPTURED: {name} | {business_type}")
        
        return f"Thanks {name}. I have captured your details. Our team will call you shortly to finish the setup."

# --- 5. ENTRYPOINT ---
async def entrypoint(ctx: JobContext):
    # Connect to the room
    await ctx.connect(auto_subscribe=True)

    # Initialize data
    userdata = SessionData()

    # Create the agent
    agent = RazorpaySDR()
    
    # Start the session
    session = AgentSession[SessionData](userdata=userdata)
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint
        )
    )