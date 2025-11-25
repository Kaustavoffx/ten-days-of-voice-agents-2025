import logging
import os
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli

load_dotenv()
logger = logging.getLogger("test-connection")
logger.setLevel(logging.INFO)

async def entrypoint(ctx: JobContext):
    logger.info("Attempting to connect to LiveKit Room...")
    await ctx.connect(auto_subscribe=True)
    logger.info("Successfully connected to LiveKit Room!")
    await ctx.room.local_participant.publish_data("Hello from Python!")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))