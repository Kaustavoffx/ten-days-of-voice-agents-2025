"""
Active Recall Coach - Multi-Agent Voice AI System
Using Murf Falcon TTS for the fastest text-to-speech generation
FIXED: Voice input now properly processes audio
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import deepgram, openai, silero, murf

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Make sure our logger is visible
logger = logging.getLogger("active-recall-coach")
logger.setLevel(logging.DEBUG)

# Also enable livekit agent logs
logging.getLogger("livekit.agents").setLevel(logging.DEBUG)
logging.getLogger("livekit.plugins").setLevel(logging.DEBUG)

# Load environment variables immediately
load_dotenv()

@dataclass
class SessionData:
    """Stores learning session data across agent transfers"""
    topic: Optional[str] = None
    difficulty_level: str = "beginner"
    concepts_covered: list[str] = field(default_factory=list)
    recall_attempts: int = 0
    correct_recalls: int = 0
    session_notes: list[str] = field(default_factory=list)
    personas: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None
    ctx: Optional[JobContext] = None

    def summarize(self) -> str:
        accuracy = (self.correct_recalls / self.recall_attempts * 100) if self.recall_attempts > 0 else 0
        return f"Topic: {self.topic}, Level: {self.difficulty_level}, Concepts: {len(self.concepts_covered)}, Accuracy: {accuracy:.1f}%"

    def get_performance_summary(self) -> str:
        accuracy = (self.correct_recalls / self.recall_attempts * 100) if self.recall_attempts > 0 else 0
        return f"""
        📊 Session Performance:
        - Topic: {self.topic}
        - Difficulty: {self.difficulty_level}
        - Concepts Covered: {len(self.concepts_covered)}
        - Recall Attempts: {self.recall_attempts}
        - Correct Recalls: {self.correct_recalls}
        - Accuracy: {accuracy:.1f}%
        """

RunContext_T = RunContext[SessionData]

class BaseAgent(Agent):
    """Base agent with shared functionality for context management"""
    
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"🎤 {agent_name} - ENTERING AND READY")

        userdata: SessionData = self.session.userdata
        if userdata.ctx and userdata.ctx.room:
            await userdata.ctx.room.local_participant.set_attributes({"agent": agent_name})

        chat_ctx = self.chat_ctx.copy()

        if userdata.prev_agent:
            items_copy = self._truncate_chat_ctx(
                userdata.prev_agent.chat_ctx.items, keep_function_call=True
            )
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in items_copy if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        chat_ctx.add_message(
            role="system",
            content=f"You are the {agent_name}. Session info: {userdata.summarize()}"
        )
        await self.update_chat_ctx(chat_ctx)
        
        # CRITICAL FIX: Force the agent to generate its first message
        logger.info(f"🔊 {agent_name} - Generating initial greeting")
        self.session.generate_reply()

    def _truncate_chat_ctx(
        self,
        items: list,
        keep_last_n_messages: int = 8,
        keep_system_message: bool = False,
        keep_function_call: bool = False,
    ) -> list:
        def _valid_item(item) -> bool:
            if not keep_system_message and item.type == "message" and item.role == "system":
                return False
            if not keep_function_call and item.type in ["function_call", "function_call_output"]:
                return False
            return True

        new_items = []
        for item in reversed(items):
            if _valid_item(item):
                new_items.append(item)
            if len(new_items) >= keep_last_n_messages:
                break
        new_items = new_items[::-1]

        while new_items and new_items[0].type in ["function_call", "function_call_output"]:
            new_items.pop(0)

        return new_items

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> Agent:
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.personas[name]
        userdata.prev_agent = current_agent
        return next_agent


class IntakeAgent(BaseAgent):
    """Initial agent that sets up the learning session"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are the Intake Agent for the Active Recall Learning Coach.
            
            **IMPORTANT: When you first enter, IMMEDIATELY greet the student! Say "Hello! I'm your Active Recall Learning Coach" and introduce yourself right away.**
            
            🎯 Your Mission:
            1. **START by warmly greeting students** - don't wait for them to speak first!
            2. Briefly explain active recall learning in one sentence
            3. Ask: "What would you like to learn about today?"
            4. Ask: "What's your current knowledge level - beginner, intermediate, or advanced?"
            5. Record both answers using the appropriate tools
            6. Transfer to the Teaching Agent once complete
            
            💡 About Active Recall:
            Active recall is a scientifically-proven learning technique where you actively 
            retrieve information from memory, which creates stronger neural pathways than 
            passive review. It's like exercising a muscle - the more you practice retrieving 
            information, the stronger your memory becomes!
            
            Keep your tone:
            - Warm and encouraging
            - Clear and concise
            - Enthusiastic about learning
            - Professional yet friendly
            
            **Remember: Greet them immediately when entering!**""",
            stt=deepgram.STT(
                model="nova-2-general",
                language="en-US"
            ),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=murf.TTS(
                model="FALCON",
                api_key=os.getenv("MURF_API_KEY")
            ),
            vad=silero.VAD.load(
                min_speech_duration=0.2,      # Detect speech quickly
                min_silence_duration=1.5,     # Wait longer before cutting off
            )
        )

    @function_tool
    async def record_topic(self, context: RunContext_T, topic: str) -> str:
        """Record the learning topic chosen by the student"""
        context.userdata.topic = topic
        context.userdata.session_notes.append(f"Topic selected: {topic}")
        logger.info(f"Topic recorded: {topic}")
        return f"Perfect! I've recorded that you want to learn about {topic}. This is an excellent choice!"

    @function_tool
    async def record_difficulty_level(self, context: RunContext_T, level: str) -> str:
        """Record the student's knowledge level (beginner, intermediate, or advanced)"""
        level = level.lower().strip()
        if level not in ["beginner", "intermediate", "advanced"]:
            return "Please specify either beginner, intermediate, or advanced."
        
        context.userdata.difficulty_level = level
        context.userdata.session_notes.append(f"Level set: {level}")
        logger.info(f"Difficulty level recorded: {level}")
        return f"Great! I've set your knowledge level as {level}."

    @function_tool
    async def transfer_to_teaching(self, context: RunContext_T) -> Agent:
        """Transfer to the Teaching Agent once topic and level are recorded"""
        userdata = context.userdata
        if not userdata.topic:
            return "I still need to know what topic you'd like to learn about."
        if not userdata.difficulty_level:
            return "I still need to know your knowledge level."
        
        await self.session.say(
            f"Excellent! You're all set to learn about {userdata.topic} at the {userdata.difficulty_level} level. "
            f"Let me transfer you to your Teaching Agent who will guide you through the learning process."
        )
        return await self._transfer_to_agent("teaching", context)


class TeachingAgent(BaseAgent):
    """Agent that teaches concepts and prepares for recall"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are the Teaching Agent in the Active Recall Learning system.
            
            **IMPORTANT: When you first enter, immediately welcome the student and start teaching!**
            
            🎓 Your Mission:
            1. Welcome the student warmly and start teaching right away
            2. Teach the student about their chosen topic at their specified level
            3. Break complex concepts into digestible 2-3 minute chunks
            4. Use clear explanations, real-world examples, and analogies
            5. After teaching key concepts, record them using add_concepts_covered
            6. Check if student wants to test their recall or learn more
            7. Transfer to Recall Testing Agent when they're ready
            
            📚 Teaching Strategies by Level:
            
            BEGINNER:
            - Start with fundamentals and basic definitions
            - Use everyday analogies and simple examples
            - Build concepts step-by-step
            - Check understanding frequently
            - Avoid jargon or explain it clearly
            
            INTERMEDIATE:
            - Assume basic knowledge, dive deeper into mechanisms
            - Connect concepts to broader context
            - Introduce some technical terminology
            - Use more sophisticated examples
            
            ADVANCED:
            - Discuss nuances, edge cases, and complexities
            - Reference current research or debates
            - Use technical terminology appropriately
            - Challenge them with thought-provoking questions
            
            💡 Teaching Best Practices:
            - Use the Feynman Technique: explain simply and clearly
            - Provide concrete examples before abstract concepts
            - Create mental anchors and memory hooks
            - Make content engaging and relatable
            - Ask comprehension questions periodically
            
            After covering 2-3 key concepts, suggest testing their recall!""",
            stt=deepgram.STT(
                model="nova-2-general",
                language="en-US"
            ),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=murf.TTS(
                model="FALCON",
                api_key=os.getenv("MURF_API_KEY")
            ),
            vad=silero.VAD.load(
                min_speech_duration=0.2,
                min_silence_duration=1.5,
            )
        )

    @function_tool
    async def add_concepts_covered(
        self, 
        context: RunContext_T, 
        concepts: list[str]
    ) -> str:
        """Add concepts that have been taught to the student"""
        context.userdata.concepts_covered.extend(concepts)
        context.userdata.session_notes.append(f"Concepts taught: {', '.join(concepts)}")
        logger.info(f"Added concepts: {concepts}")
        return f"I've recorded that we covered: {', '.join(concepts)}. These are important building blocks!"

    @function_tool
    async def transfer_to_recall(self, context: RunContext_T) -> Agent:
        """Transfer to Recall Testing Agent when student is ready to practice"""
        userdata = context.userdata
        if len(userdata.concepts_covered) < 2:
            return "Let me teach you at least 2-3 concepts before we test your recall."
        
        await self.session.say(
            "Excellent! Now comes the exciting part - testing your recall. "
            "Remember, struggling to remember is actually beneficial - it strengthens your memory! "
            "Let me transfer you to the Recall Testing Agent."
        )
        return await self._transfer_to_agent("recall", context)

    @function_tool
    async def transfer_to_intake(self, context: RunContext_T) -> Agent:
        """Transfer back to Intake if student wants to learn a different topic"""
        await self.session.say(
            "No problem! Let me transfer you back to start a fresh learning session with a new topic."
        )
        # Reset for new session
        context.userdata.topic = None
        context.userdata.concepts_covered = []
        context.userdata.recall_attempts = 0
        context.userdata.correct_recalls = 0
        return await self._transfer_to_agent("intake", context)


class RecallTestingAgent(BaseAgent):
    """Agent that tests student's recall and provides feedback"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are the Recall Testing Agent in the Active Recall Learning system.
            
            **IMPORTANT: When you first enter, immediately welcome them and start testing!**
            
            🧪 Your Mission:
            1. Welcome them warmly and immediately ask your first recall question
            2. Test the student's ability to recall what they learned
            3. Ask open-ended questions WITHOUT giving hints initially
            4. Listen carefully to their complete explanations
            5. Evaluate their understanding (not word-perfect memorization)
            6. Record each attempt as correct or incorrect
            7. Provide immediate, constructive feedback
            8. After 3-5 questions, transfer to Feedback Agent
            
            ❓ Question Techniques:
            - "Can you explain what [concept] means in your own words?"
            - "How does [concept] work?"
            - "What's the difference between [A] and [B]?"
            - "Why is [concept] important?"
            - "Can you give me an example of [concept]?"
            
            ✅ Evaluation Guidelines:
            - Focus on conceptual understanding, not exact wording
            - Partial understanding counts as partially correct
            - Look for: key ideas, relationships, practical application
            - Don't expect perfection - learning is a process!
            
            💬 Feedback Approach:
            - If CORRECT: "Excellent! You've captured the key idea..."
            - If PARTIAL: "Good start! You mentioned X, but let me clarify Y..."
            - If INCORRECT: "I can see why you'd think that. Actually..."
            - Always end with encouragement and correct information
            
            🎯 Remember:
            - Difficulty recalling is NORMAL and BENEFICIAL
            - The struggle strengthens memory formation
            - Mistakes reveal gaps that need attention
            - Each attempt improves retention
            
            After 3-5 recall questions, transfer to the Feedback Agent for summary.""",
            stt=deepgram.STT(
                model="nova-2-general",
                language="en-US"
            ),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=murf.TTS(
                model="FALCON",
                api_key=os.getenv("MURF_API_KEY")
            ),
            vad=silero.VAD.load(
                min_speech_duration=0.2,
                min_silence_duration=1.5,
            )
        )

    @function_tool
    async def record_recall_attempt(
        self, 
        context: RunContext_T, 
        was_correct: bool,
        concept_tested: str
    ) -> str:
        """Record whether the student's recall attempt was correct"""
        context.userdata.recall_attempts += 1
        if was_correct:
            context.userdata.correct_recalls += 1
        
        accuracy = (context.userdata.correct_recalls / context.userdata.recall_attempts * 100)
        context.userdata.session_notes.append(
            f"Recall #{context.userdata.recall_attempts}: {concept_tested} - {'✓' if was_correct else '✗'}"
        )
        logger.info(f"Recall attempt recorded. Concept: {concept_tested}, Correct: {was_correct}, Accuracy: {accuracy:.1f}%")
        return f"Recorded. Your current recall accuracy is {accuracy:.1f}%"

    @function_tool
    async def transfer_to_teaching(self, context: RunContext_T) -> Agent:
        """Transfer back to Teaching Agent if concepts need reinforcement"""
        await self.session.say(
            "I can see some concepts would benefit from more review. "
            "Let me transfer you back to the Teaching Agent for reinforcement and deeper understanding."
        )
        return await self._transfer_to_agent("teaching", context)

    @function_tool
    async def transfer_to_feedback(self, context: RunContext_T) -> Agent:
        """Transfer to Feedback Agent for session summary and next steps"""
        userdata = context.userdata
        if userdata.recall_attempts < 3:
            return "Let's test at least 3 concepts before wrapping up."
        
        await self.session.say(
            "Great work on the recall practice! You've tested yourself on multiple concepts. "
            "Let me transfer you to the Feedback Agent who will provide a comprehensive session summary."
        )
        return await self._transfer_to_agent("feedback", context)


class FeedbackAgent(BaseAgent):
    """Agent that provides session summary and learning recommendations"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are the Feedback Agent in the Active Recall Learning system.
            
            **IMPORTANT: When you first enter, immediately provide an encouraging summary of their session!**
            
            🎉 Your Mission:
            1. Start immediately with a warm, encouraging summary
            2. Provide a comprehensive session summary
            3. Highlight specific concepts learned and recall performance
            4. Celebrate successes and progress made
            5. Identify areas for improvement without discouragement
            6. Explain the science behind active recall
            7. Suggest optimal next steps for the student
            
            📊 Summary Components:
            - Topic covered and difficulty level
            - Number of concepts learned
            - Recall accuracy percentage
            - Specific strengths shown
            - Areas for further practice
            - Learning insights
            
            🎯 Feedback Principles:
            - Always be genuinely positive and encouraging
            - Frame mistakes as valuable learning opportunities
            - Use specific examples from their session
            - Emphasize the process, not just the outcome
            - Remind them that struggle = growth
            
            🧠 Active Recall Science to Share:
            - Testing yourself is 2-3x more effective than re-reading
            - The retrieval effort strengthens neural pathways
            - Mistakes during recall enhance learning (desirable difficulty)
            - Spacing practice sessions improves long-term retention
            - Active recall builds confidence and reduces test anxiety
            
            📅 Recommend Spaced Repetition:
            - Review in 1 day (50% retention boost)
            - Review in 3 days (another 30% boost)
            - Review in 1 week (locks in long-term memory)
            
            🚀 Next Steps Options:
            1. Continue with same topic (go deeper)
            2. Start new topic (breadth)
            3. End session (come back later)
            
            Make your feedback specific, actionable, and motivating!""",
            stt=deepgram.STT(
                model="nova-2-general",
                language="en-US"
            ),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=murf.TTS(
                model="FALCON",
                api_key=os.getenv("MURF_API_KEY")
            ),
            vad=silero.VAD.load(
                min_speech_duration=0.2,
                min_silence_duration=1.5,
            )
        )

    @function_tool
    async def transfer_to_teaching(self, context: RunContext_T) -> Agent:
        """Continue learning the same topic with more depth"""
        await self.session.say(
            "Excellent choice! Let's dive even deeper into this topic. "
            "I'm transferring you back to the Teaching Agent for more advanced concepts."
        )
        return await self._transfer_to_agent("teaching", context)

    @function_tool
    async def transfer_to_intake(self, context: RunContext_T) -> Agent:
        """Start a new learning session with a different topic"""
        await self.session.say(
            "Great! Let's start fresh with a new topic. "
            "Remember to review today's material in a day or two for best retention. "
            "Transferring you to the Intake Agent now."
        )
        # Reset session data for new topic
        context.userdata.topic = None
        context.userdata.difficulty_level = "beginner"
        context.userdata.concepts_covered = []
        context.userdata.recall_attempts = 0
        context.userdata.correct_recalls = 0
        context.userdata.session_notes = []
        
        return await self._transfer_to_agent("intake", context)

    @function_tool
    async def end_session(self, context: RunContext_T) -> str:
        """End the learning session with final summary"""
        userdata = context.userdata
        accuracy = (userdata.correct_recalls / userdata.recall_attempts * 100) if userdata.recall_attempts > 0 else 0
        
        summary = f"""
🎓 Thank you for using the Active Recall Coach powered by Murf Falcon TTS!

📊 Final Session Summary:
━━━━━━━━━━━━━━━━━━━━━
• Topic: {userdata.topic}
• Difficulty Level: {userdata.difficulty_level}
• Concepts Learned: {len(userdata.concepts_covered)}
• Recall Tests: {userdata.recall_attempts}
• Accuracy: {accuracy:.1f}%

✨ Key Takeaways:
The active recall method you practiced today is scientifically proven to be 2-3 times 
more effective than passive review. Each time you struggled to remember something, 
you were actually strengthening those neural pathways!

📅 For Best Results:
• Review this material again in 1 day
• Test yourself again in 3 days 
• Final review in 1 week

This spacing schedule will lock the information into your long-term memory.

🚀 You're building a powerful learning skill that will serve you for life. 
Keep practicing active recall, and watch your learning accelerate!

Happy learning! 🌟
        """
        
        await self.session.say(summary)
        logger.info(f"Session ended. {userdata.get_performance_summary()}")
        return "Session ended successfully"


async def entrypoint(ctx: JobContext):
    """Main entry point for the Active Recall Coach system"""
    
    # Connect to the room explicitly using .env settings
    logger.info("🔌 Connecting to LiveKit room...")
    await ctx.connect(auto_subscribe=True)
    logger.info("✅ Connected to room successfully")

    logger.info("🚀 Starting Active Recall Coach with Murf Falcon TTS")
    
    # Initialize session data
    userdata = SessionData(ctx=ctx)
    
    # Create all specialized agents
    intake_agent = IntakeAgent()
    teaching_agent = TeachingAgent()
    recall_agent = RecallTestingAgent()
    feedback_agent = FeedbackAgent()

    # Register all agents for cross-agent transfers
    userdata.personas.update({
        "intake": intake_agent,
        "teaching": teaching_agent,
        "recall": recall_agent,
        "feedback": feedback_agent
    })

    # Create the session with typed userdata
    session = AgentSession[SessionData](userdata=userdata)

    # Start with the Intake Agent
    logger.info("🎬 Starting session with Intake Agent - should greet immediately")
    await session.start(
        agent=intake_agent,
        room=ctx.room,
    )
    
    logger.info("✨ Active Recall Coach session started - Agent should be speaking and listening!")


if __name__ == "__main__":
    # Load vars again to be sure
    load_dotenv()
    
    livekit_url = os.getenv("LIVEKIT_URL")
    livekit_api_key = os.getenv("LIVEKIT_API_KEY")
    livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")

    # Run the worker with explicit connection details
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=livekit_url,
            api_key=livekit_api_key,
            api_secret=livekit_api_secret
        )
    )