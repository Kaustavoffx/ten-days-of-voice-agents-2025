import logging
import os
import json
from typing import Annotated
from dotenv import load_dotenv

from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
# CHANGED: Switched from google to openai
from livekit.plugins import openai, silero, murf

load_dotenv()
logger = logging.getLogger("shop-agent")
logger.setLevel(logging.INFO)

# Load Catalog
with open("catalog.json", "r") as f:
    CATALOG = json.load(f)

# --- SESSION STATE ---
class SessionData:
    cart: list = []

RunContext_T = RunContext[SessionData]

# --- THE AGENT ---
class ShopAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are 'ShopBot', a helpful e-commerce voice assistant.
            
            GOAL: Help customers find products and manage their cart.
            
            CATALOG:
            """ + json.dumps(CATALOG) + """
            
            RULES:
            1. Keep answers short and friendly.
            2. If they ask for a product, check the catalog.
            3. Use 'add_to_cart' to add items.
            4. Use 'checkout' to finish.
            
            Be enthusiastic!""",
            
            # HEARING: OpenAI Whisper (More robust than Deepgram on hotspots)
            stt=openai.STT(),
            
            # BRAIN: OpenAI GPT-4o Mini (Fast & Reliable)
            llm=openai.LLM(model="gpt-4o-mini"),
            
            # VOICE: Murf Falcon (Challenge Requirement)
            tts=murf.TTS(
                model="FALCON",
                api_key=os.getenv("MURF_API_KEY")
            ),
            
            vad=silero.VAD.load(min_speech_duration=0.1)
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello! Welcome to the AI Store. How can I help you shop today?")

    # --- TOOLS ---
    @function_tool
    async def browse_products(self, context: RunContext_T, category: Annotated[str, "Optional category filter"] = None) -> str:
        """List available products."""
        items = CATALOG
        if category:
            items = [i for i in CATALOG if category.lower() in i["category"].lower()]
        
        if not items: return "No products found."
        return f"Available: {', '.join([i['name'] + ' (' + str(i['price']) + ')' for i in items])}"

    @function_tool
    async def add_to_cart(self, context: RunContext_T, product_name: str, quantity: int) -> str:
        """Add item to cart."""
        for item in CATALOG:
            if product_name.lower() in item["name"].lower():
                context.userdata.cart.append({"item": item["name"], "qty": quantity, "price": item["price"]})
                return f"Added {quantity} {item['name']} to cart."
        return "Product not found."

    @function_tool
    async def checkout(self, context: RunContext_T) -> str:
        """Finalize order."""
        if not context.userdata.cart: return "Cart is empty."
        total = sum([i['qty'] * i['price'] for i in context.userdata.cart])
        order_summary = ", ".join([f"{i['qty']}x {i['item']}" for i in context.userdata.cart])
        
        # Clear cart after order
        context.userdata.cart = [] 
        
        return f"Order Confirmed! You bought: {order_summary}. Total: {total} Rupees."

# --- ENTRYPOINT ---
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=True)
    userdata = SessionData()
    agent = ShopAgent()
    session = AgentSession(userdata=userdata)
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))