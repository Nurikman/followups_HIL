import sys
import os
import logging
from typing import List

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from dataclasses import dataclass
from pydantic_ai import RunContext, Agent
from pydantic import BaseModel

from openai_model import get_openai_model
from dotenv import load_dotenv

load_dotenv()

# Set up basic logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


class ConversationSegment(BaseModel):
    segment_id: int
    topic: str
    tone: str
    conversation_direction: str
    interaction_type: str  # "useful_interaction" or "personal_interaction"
    start_line: int  # Starting line number in conversation (0-based)
    end_line: int    # Ending line number in conversation (0-based, inclusive)
    content: str = ""  # Will be populated programmatically
    engagement_score: int  # 1-10 scale
    engagement_justification: str
    enjoyment_score: int   # 1-10 scale
    enjoyment_justification: str
    combined_score: int    # Sum of engagement + enjoyment


class SegmenterRaterResult(BaseModel):
    segments: List[ConversationSegment]


@dataclass
class SegmenterRaterDeps:
    conversation: str  # Single conversation text


def populate_segment_content(segments: List[ConversationSegment], conversation: str) -> List[ConversationSegment]:
    """Programmatically populate the content field of segments based on start_line and end_line."""
    conversation_lines = conversation.strip().split('\n')
    
    for segment in segments:
        # Validate line numbers
        if segment.start_line < 0 or segment.start_line >= len(conversation_lines):
            logger.warning(f"Invalid start_line {segment.start_line} for segment {segment.segment_id}")
            segment.content = "Invalid segment boundaries"
            continue
            
        if segment.end_line < 0 or segment.end_line >= len(conversation_lines):
            logger.warning(f"Invalid end_line {segment.end_line} for segment {segment.segment_id}, adjusting to {len(conversation_lines)-1}")
            segment.end_line = len(conversation_lines) - 1
            
        if segment.start_line > segment.end_line:
            logger.warning(f"Invalid segment {segment.segment_id}: start_line {segment.start_line} > end_line {segment.end_line}")
            segment.content = "Invalid segment boundaries"
            continue
        
        # Extract content from conversation lines
        segment_lines = conversation_lines[segment.start_line:segment.end_line + 1]
        segment.content = '\n'.join(segment_lines)
        
        logger.info(f"Populated content for segment {segment.segment_id}: lines {segment.start_line}-{segment.end_line}")
    
    return segments


def make_agent_chat_segmenter_rater(model_name="gpt-4o"):
    agent = Agent(
        get_openai_model(model_name),
        deps_type=SegmenterRaterDeps,
        retries=3,
        result_type=SegmenterRaterResult,
    )
    
    @agent.system_prompt
    def system_prompt(ctx: RunContext[SegmenterRaterDeps]) -> str:
        conversation_lines = ctx.deps.conversation.strip().split('\n')
        total_lines = len(conversation_lines)
        
        # Add line numbers to conversation for reference
        numbered_conversation = ""
        for i, line in enumerate(conversation_lines):
            numbered_conversation += f"Line {i}: {line}\n"
        
        return f"""
        You are an expert conversation analyzer that segments dialogues and rates user engagement.
        Your task is to analyze a conversation and create a structured breakdown with engagement ratings.
        
        LANGUAGE DETECTION: First, analyze the language of the input conversation. If the conversation is primarily in Russian, provide ALL your analysis (topic, tone, conversation_direction, interaction_type, engagement_justification, enjoyment_justification) in Russian. If the conversation is in English or other languages, respond in English.
        
        Conversation to analyze (with line numbers):
        
        {numbered_conversation}
        
        IMPORTANT: This conversation has {total_lines} lines (numbered 0 to {total_lines-1}).
        You must specify start_line and end_line for each segment. Content will be extracted programmatically.
        
        Please analyze this dialogue and create a structured breakdown:
        
        SEGMENTATION STRATEGY:
        - Create segments based on meaningful conversation events and topic shifts
        - Each segment should represent a distinct conversation moment or theme
        - Look for natural breakpoints: topic changes, emotional shifts, new questions/problems
        - Segment granularly - better to have more meaningful segments than fewer generic ones
        - Each segment should be substantial enough to analyze (minimum 2-3 exchanges)
        
        SEGMENT CRITERIA:
        1. Topic shift: When conversation moves to a new subject
        2. Emotional change: When user's mood/enthusiasm changes
        3. Problem-solution cycles: Each user question/problem and its resolution
        4. Conversation direction change: From learning to planning, casual to serious, etc.
        5. Time gaps: If there are clear temporal breaks in conversation
        6. Engagement level shifts: When user becomes more/less engaged
        
        For each segment, analyze:
        1. Topic: What specific aspect is being discussed
        2. Tone: Emotional quality and communication style
        3. Conversation direction: Where this part of the conversation is heading
        4. Interaction type: Classify the nature of the interaction
        5. Start_line: First line number of the segment (0-based)
        6. End_line: Last line number of the segment (0-based, inclusive)
        7. Engagement score (1-10): How actively the user participates
        8. Enjoyment score (1-10): How much the user seems to enjoy this part
        
        CRITICAL: start_line and end_line must be between 0 and {total_lines-1}.
        DO NOT specify content - it will be extracted programmatically based on line numbers.
        
        Interaction Type Classification:
        - "useful_interaction": User is using the agent as a tool or service
          Examples: Learning (languages, skills, subjects), asking for information from internet, 
          seeking professional advice, requesting weather updates, playing games, learning history,
          getting help with tasks, troubleshooting, research, educational content, tutorials
        - "personal_interaction": User is sharing personal experiences or seeking personal support
          Examples: Sharing illness/health updates, announcing good/bad news, asking for relationship advice,
          discussing personal problems, sharing emotions/feelings, seeking emotional support,
          talking about family/friends, personal achievements, life events
        
        Rating Guidelines:
        - Engagement (1-10): Question frequency, response depth, follow-up interest, active participation
        - Enjoyment (1-10): Positive sentiment, enthusiasm, satisfaction expressions, emotional investment
        - Look for indicators: exclamation marks, positive words, curiosity, gratitude, humor
        
        OUTPUT REQUIREMENTS:
        - Create comprehensive segmentation covering the entire conversation
        - Each segment must have substantial content (not just single exchanges)
        - Specify accurate start_line and end_line for each segment
        - Provide detailed justifications for engagement and enjoyment scores
        - Calculate combined score (engagement + enjoyment) for ranking
        - Sort segments by combined score (engagement + enjoyment) in descending order
        
        Focus on capturing the richness and variety of the user's conversation experience.
        """
    
    return agent 