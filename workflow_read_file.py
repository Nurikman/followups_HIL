from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import logging

import sys
import os

# Import agents directly
from agents.chat_segmenter_rater import (
    make_agent_chat_segmenter_rater,
    SegmenterRaterDeps,
    ConversationSegment
)
from agents.conversation_starter_generator import (
    make_agent_conversation_starter_generator,
    StarterGeneratorDeps,
    ConversationStarter
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Add console handler to see logging output
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
logger.addHandler(handler)


class ConversationFollowUpState(TypedDict):
    user_id: str
    raw_conversations: List[str]
    conversation_segments: List[ConversationSegment]
    top_segments: List[ConversationSegment]
    conversation_starters: List[ConversationStarter]


def get_conversation_followup_workflow(model_name="gpt-4o"):
    """Creates the conversation follow-up workflow with 2 agents"""
    
    agent_segmenter_rater = make_agent_chat_segmenter_rater(model_name=model_name)
    agent_starter_generator = make_agent_conversation_starter_generator(model_name=model_name)

    async def node_segment_and_rate(state: ConversationFollowUpState):
        """Node 1: Segment conversations and rate engagement"""
        logger.info("Running conversation segmentation and rating")

        # Process each conversation individually
        all_segments = []
        for i, conversation in enumerate(state["raw_conversations"]):
            logger.info(f"Processing conversation {i+1}/{len(state['raw_conversations'])}")
            
            deps = SegmenterRaterDeps(
                conversation=conversation
            )

            result = await agent_segmenter_rater.run("Please analyze this conversation.", deps=deps)
            segments = result.data.segments
            all_segments.extend(segments)

        logger.info(f"Created {len(all_segments)} conversation segments total")

        # Get top 3 segments by combined score (engagement + enjoyment) for better coverage
        top_segments = sorted(all_segments, key=lambda x: x.combined_score, reverse=True)[:3]
        
        return {
            "conversation_segments": all_segments,
            "top_segments": top_segments
        }

    async def node_generate_starters(state: ConversationFollowUpState):
        """Node 2: Generate conversation starters from top segments"""
        logger.info("Generating conversation starters")

        deps = StarterGeneratorDeps(
            top_segments=state["top_segments"]
        )

        result = await agent_starter_generator.run(deps=deps)
        starters = result.data

        logger.info(f"Generated {len(starters)} conversation starters")

        return {"conversation_starters": starters}

    async def node_finish(state: ConversationFollowUpState):
        """Final node: Display results"""
        print("\n" + "="*60)
        print("📊 FINAL RESULTS")
        print("="*60)
        print(f"👤 User ID: {state['user_id']}")
        print(f"📝 Processed {len(state['raw_conversations'])} conversations")
        print(f"🔍 Generated {len(state['conversation_segments'])} total segments")
        
        print("\n📈 TOP CONVERSATION SEGMENTS:")
        print("-" * 60)
        for i, segment in enumerate(state["top_segments"], 1):
            print(f"\n{i}. SEGMENT {segment.segment_id} (Score: {segment.combined_score}/20)")
            print(f"   📋 Topic: {segment.topic}")
            print(f"   🎯 Engagement: {segment.engagement_score}/10 - {segment.engagement_justification}")
            print(f"   😊 Enjoyment: {segment.enjoyment_score}/10 - {segment.enjoyment_justification}")
            print(f"   💬 Content: {segment.content[:150]}...")
            
            # Show more detail for top 5 segments
            if i <= 5:
                print(f"   🎭 Tone: {segment.tone}")
                print(f"   🧭 Direction: {segment.conversation_direction}")
        
        print("\n🎯 GENERATED CONVERSATION STARTERS:")
        print("-" * 60)
        # Show top 15 starters (instead of all 30 to keep output manageable)
        for starter in state["conversation_starters"][:15]:
            print(f"\n{starter.rank}. {starter.starter}")
            print(f"   💡 Context: {starter.context}")
            
        if len(state["conversation_starters"]) > 15:
            print(f"\n... and {len(state['conversation_starters']) - 15} more conversation starters (ranks 16-30)")

        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        
        return state

    # Build the workflow graph
    builder = StateGraph(ConversationFollowUpState)

    # Add nodes
    builder.add_node("node_segment_and_rate", node_segment_and_rate)
    builder.add_node("node_generate_starters", node_generate_starters)
    builder.add_node("node_finish", node_finish)

    # Add edges (sequential flow)
    builder.add_edge(START, "node_segment_and_rate")
    builder.add_edge("node_segment_and_rate", "node_generate_starters")
    builder.add_edge("node_generate_starters", "node_finish")
    builder.add_edge("node_finish", END)

    # Compile with memory
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    return graph


# Helper function to read conversations from chat log file
def read_conversations_from_file(file_path: str) -> List[str]:
    """
    Read conversations from a chat log text file.
    Expected format: Chat log with timestamps and Agent/User messages.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        # Parse chat log format and convert to conversation format
        conversation_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Extract message content from chat log format
            # Format: [timestamp] Agent: message or [timestamp] User 2: message
            if '] Agent:' in line:
                # Extract agent message
                message_part = line.split('] Agent:', 1)[1].strip()
                conversation_lines.append(f"agent: {message_part}")
            elif '] User 2:' in line:
                # Extract user message  
                message_part = line.split('] User 2:', 1)[1].strip()
                conversation_lines.append(f"user: {message_part}")
        
        # Join all lines into one conversation
        if conversation_lines:
            full_conversation = '\n'.join(conversation_lines)
            conversations = [full_conversation]  # Return as single conversation
            
            logger.info(f"Successfully parsed chat log with {len(conversation_lines)} messages from {file_path}")
            return conversations
        else:
            logger.warning(f"No valid chat messages found in {file_path}")
            return []
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return []


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_workflow():
        print("🚀 Starting Intelligent Conversation Follow-Up System")
        print("=" * 60)
        
        # Read conversations from chat log file
        user_id = "file_user"  # Default user ID for file-based processing
        conversation_file = "user_conversations/conv-User-2025-07-03-translated.txt"  # Specific filename
        print(f"🔍 Reading conversations from file: {conversation_file}")
        
        conversations = read_conversations_from_file(conversation_file)
        
        if not conversations:
            print("❌ No conversations found in file. Using sample data for testing.")
            # Fallback to sample data if no file or empty file
            conversations = [
                "user: What is a turnpike? I've been wondering about this for a while.\nagent: A turnpike is a toll road, especially in the northeastern United States. They're called turnpikes because historically, they had gates that would turn to let traffic through after paying the toll.\nuser: Oh that's really interesting! I never knew the etymology. What about blind spots when driving?\nagent: Blind spots are areas around your vehicle that you cannot see in your mirrors or through your windows directly. They're typically located to the sides and rear of your vehicle.\nuser: Got it, that's very helpful! I'm learning so much about driving today."
            ]
        
        print(f"📝 Processing {len(conversations)} conversation(s)")
        print("=" * 60)
        
        workflow = get_conversation_followup_workflow()
        
        config = {
            "configurable": {"thread_id": f"file-{user_id}"}
        }
        
        initial_state = ConversationFollowUpState(
            user_id=user_id,
            raw_conversations=conversations,
            conversation_segments=[],
            top_segments=[],
            conversation_starters=[]
        )
        
        print("🤖 Running AI workflow...")
        result = await workflow.ainvoke(initial_state, config=config)
        
        print("\n" + "=" * 60)
        print("✅ Workflow completed successfully!")
        print("=" * 60)
        
        return result
    
    # Run test with file data
    asyncio.run(test_workflow()) 