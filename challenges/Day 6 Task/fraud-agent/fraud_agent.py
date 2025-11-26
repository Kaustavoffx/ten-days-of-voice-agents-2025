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
from database_helper import get_user, get_flagged_txn, update_txn_status

load_dotenv()
logger = logging.getLogger("fraud-agent")
logger.setLevel(logging.INFO)

# --- SESSION STATE ---
class SessionData:
    verified_user_id: str = None

RunContext_T = RunContext[SessionData]

# --- THE AGENT CLASS ---
class FraudDetectionAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are SecureGuard, a Fraud Detection Agent for HDFC Bank.
            Your job is to verify a high-value suspicious transaction.
            
            SCRIPT & RULES:
            1. Start by greeting and asking for the User ID.
            2. Use the 'verify_identity' tool to check if they exist.
            3. Once verified, use 'check_suspicious_activity' to find the transaction.
            4. Tell them: 'I see a transaction for [Amount] at [Merchant] in [Location]. Did you do this?'
            5. If NO: Use 'process_transaction' with 'block'.
            6. If YES: Use 'process_transaction' with 'approve'.
            
            Keep responses short, professional, and serious.""",
            
            # Using OpenAI STT (Network safe)
            stt=openai.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            # Murf Falcon TTS (Challenge Requirement)
            tts=murf.TTS(
                model="FALCON",
                api_key=os.getenv("MURF_API_KEY")
            ),
            vad=silero.VAD.load(min_speech_duration=0.1)
        )

    # --- TOOLS ---
    @function_tool
    async def verify_identity(self, context: RunContext_T, user_id: Annotated[str, "The User ID provided by customer"]) -> str:
        """Verify the user exists in the bank database."""
        logger.info(f"Verifying: {user_id}")
        user = get_user(user_id)
        if user:
            context.userdata.verified_user_id = user_id
            return f"Identity Verified. Name: {user['name']}. Account ending in: {user['account_number'][-4:]}."
        return "User ID not found in our database."

    @function_tool
    async def check_suspicious_activity(self, context: RunContext_T) -> str:
        """Check if the verified user has any flagged transactions."""
        if not context.userdata.verified_user_id:
            return "Error: Please verify identity first."
            
        txn = get_flagged_txn(context.userdata.verified_user_id)
        if txn:
            return f"ALERT: Suspicious transaction found! ID: {txn['transaction_id']}, Amount: {txn['amount']}, Merchant: {txn['merchant']}, Location: {txn['location']}."
        return "No suspicious activity found."

    @function_tool
    async def process_transaction(
        self,
        context: RunContext_T, 
        transaction_id: Annotated[str, "The transaction ID"],
        decision: Annotated[str, "Decision: 'block' or 'approve'"]
    ) -> str:
        """Block or Approve the transaction based on user input."""
        update_txn_status(transaction_id, decision)
        if decision == "block":
            return f"Transaction {transaction_id} has been BLOCKED immediately. A fraud report has been filed."
        return f"Transaction {transaction_id} has been APPROVED. Thank you for verifying this purchase."


# --- ENTRYPOINT ---
async def entrypoint(ctx: JobContext):
    # Connect to LiveKit
    await ctx.connect(auto_subscribe=True)
    
    # Create Session Data
    userdata = SessionData()

    # Create the Agent
    agent = FraudDetectionAgent()

    # Start Session
    session = AgentSession[SessionData](userdata=userdata)
    await session.start(agent=agent, room=ctx.room)
    
    # Initial Greeting
    await session.say("Hello. This is HDFC Bank Security calling. I need to verify a recent transaction. Can you please state your User ID?")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))