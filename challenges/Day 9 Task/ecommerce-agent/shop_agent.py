import logging
import os
import json
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
            # UPDATED INSTRUCTIONS FOR GROCERY CONTEXT
            instructions="""You are 'Grocer', a helpful Quick Commerce assistant (like Zepto/Blinkit).
            
            GOAL: Help customers order groceries and snacks.
            
            CATALOG:
            """ + json.dumps(CATALOG) + """
            
            RULES:
            1. Keep answers short (1 sentence).
            2. If they ask for an item, check if it's in the catalog.
            3. Use 'add_to_cart' to add items.
            4. Use 'checkout' to finish.
            
            Be enthusiastic about food!""",
            
            stt=openai.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=murf.TTS(
                model="FALCON",
                api_key=os.getenv("MURF_API_KEY")
            ),
            vad=silero.VAD.load(min_speech_duration=0.1)
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi! Welcome to QuickMart. I can help you with groceries. What do you need?")

    # --- TOOLS ---
    @function_tool
    async def browse_products(self, context: RunContext_T, category: Annotated[str, "Optional category filter"] = None) -> str:
        """List available products."""
        items = CATALOG
        if category:
            items = [i for i in CATALOG if category.lower() in i["category"].lower()]
        
        if not items: return "Sorry, we don't have that category."
        return f"Available: {', '.join([i['name'] + ' (' + str(i['price']) + ')' for i in items])}"

    @function_tool
    async def add_to_cart(self, context: RunContext_T, product_name: str, quantity: int) -> str:
        """Add item to cart."""
        for item in CATALOG:
            # Fuzzy match (e.g. "Milk" matches "Fresh Milk")
            if product_name.lower() in item["name"].lower():
                context.userdata.cart.append({"item": item["name"], "qty": quantity, "price": item["price"]})
                return f"Added {quantity} {item['name']} to cart."
        return f"Sorry, we don't have {product_name} in stock."

    @function_tool
    async def checkout(self, context: RunContext_T) -> str:
        """Finalize order."""
        if not context.userdata.cart: return "Cart is empty."
        total = sum([i['qty'] * i['price'] for i in context.userdata.cart])
        order_summary = ", ".join([f"{i['qty']}x {i['item']}" for i in context.userdata.cart])
        
        context.userdata.cart = [] # Clear
        return f"Order Confirmed! items: {order_summary}. Total Bill: {total} Rupees. Arriving in 10 minutes!"

# --- ENTRYPOINT ---
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=True)
    userdata = SessionData()
    agent = ShopAgent()
    session = AgentSession(userdata=userdata)
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))