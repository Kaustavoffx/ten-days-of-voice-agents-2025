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
from livekit.plugins import deepgram, silero, openai, murf

load_dotenv()
logger = logging.getLogger("grocery-agent")
logger.setLevel(logging.INFO)

# Load Catalog
with open("catalog.json", "r") as f:
    CATALOG = json.load(f)

# --- SESSION DATA ---
class SessionData:
    cart: list = []

RunContext_T = RunContext[SessionData]

# --- AGENT CLASS ---
class GroceryAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are 'Grocer', a helpful AI assistant for Zepto/Blinkit.
            
            GOAL: Help the user order groceries.
            
            CATALOG:
            """ + json.dumps(CATALOG) + """
            
            RULES:
            1. Keep answers short (1 sentence).
            2. If they ask for an item, check if it's in the catalog.
            3. If yes, add it to cart using 'add_to_cart'.
            4. If they want to finish, use 'place_order'.
            5. Be friendly and fast.""",
            
            # STT: OpenAI (Better for your network)
            stt=openai.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            # TTS: Murf Falcon (Challenge Requirement)
            tts=murf.TTS(
                model="FALCON",
                api_key=os.getenv("MURF_API_KEY")
            ),
            vad=silero.VAD.load(min_speech_duration=0.1)
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi! Welcome to Zepto Voice. What groceries do you need today?")

    # --- TOOLS ---
    @function_tool
    async def add_to_cart(self, context: RunContext_T, item_name: Annotated[str, "Item Name"], quantity: int) -> str:
        """Add an item to the shopping cart."""
        # Simple check
        for item in CATALOG["items"]:
            if item_name.lower() in item["name"].lower():
                context.userdata.cart.append({"item": item["name"], "qty": quantity, "price": item["price"]})
                return f"Added {quantity} {item['name']} to cart."
        return f"Sorry, {item_name} is not in stock."

    @function_tool
    async def view_cart(self, context: RunContext_T) -> str:
        """Check what is in the cart."""
        if not context.userdata.cart:
            return "Your cart is empty."
        items = [f"{i['qty']}x {i['item']}" for i in context.userdata.cart]
        return f"You have: {', '.join(items)}."

    @function_tool
    async def place_order(self, context: RunContext_T) -> str:
        """Finalize the order."""
        if not context.userdata.cart:
            return "Cart is empty!"
        total = sum([i['qty'] * i['price'] for i in context.userdata.cart])
        # Clear cart
        context.userdata.cart = []
        return f"Order placed! Total amount is {total} Rupees. It will arrive in 10 minutes."

# --- ENTRYPOINT ---
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=True)
    userdata = SessionData()
    agent = GroceryAgent()
    session = AgentSession[SessionData](userdata=userdata)
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))