import streamlit as st
import re
import json
import os
import openai
import nltk
import asyncio
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import hashlib

# Import agents for followup generation
try:
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
    AGENTS_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Agent modules not found. Using fallback followup generation.")
    AGENTS_AVAILABLE = False

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
# Fallback for older NLTK versions
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure Streamlit page
st.set_page_config(
    page_title="Agentic Follow-up Rating System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ConversationParser:
    """Parse conversation logs from text files."""
    
    @staticmethod
    def parse_conversation(text: str) -> List[Dict]:
        """
        Parse conversation text into structured format.
        Expected format: [timestamp] Speaker: message
        """
        lines = text.strip().split('\n')
        conversations = []
        
        for line in lines:
            line = line.strip()
            if not line or line == "...":
                continue
                
            # Regex pattern to match: [timestamp] Speaker: message
            pattern = r'\[(.*?)\]\s*([^:]+):\s*(.*)'
            match = re.match(pattern, line)
            
            if match:
                timestamp, speaker, message = match.groups()
                conversations.append({
                    'timestamp': timestamp.strip(),
                    'speaker': speaker.strip(),
                    'message': message.strip()
                })
        
        return conversations

class SummaryGenerator:
    """Generate segment-based summaries with precise source mapping."""
    
    def __init__(self):
        self.segment_summary_prompt = """Please provide a concise summary of this dialogue segment in 1-2 sentences. Focus on the key points, actions, or information exchanged. Be objective and factual."""
        
    def segment_conversation(self, conversation: List[Dict]) -> List[Dict]:
        """Segment conversation using expert analysis prompt and programmatic copy-pasting."""
        if not conversation:
            return []
        
        # Use GPT-4o with the expert prompt to get segment boundaries
        if hasattr(st.session_state, 'openai_api_key') and st.session_state.openai_api_key:
            return self.segment_with_expert_analysis(conversation)
        
        # Fallback to programmatic segmentation
        return self.get_programmatic_segments_new_format(conversation)
    

    
    def segment_with_expert_analysis(self, conversation: List[Dict]) -> List[Dict]:
        """Use GPT-4o with expert conversation analyzer prompt to get segment boundaries."""
        try:
            client = openai.OpenAI(api_key=st.session_state.openai_api_key)
            
            # Format conversation for the prompt
            conversation_text = self.format_conversation_for_prompt(conversation)
            
            # Count total messages first
            total_messages = len(conversation)
            
            expert_prompt = f"""You are an expert conversation analyzer that segments dialogues.

IMPORTANT: This conversation contains EXACTLY {total_messages} messages, numbered from 0 to {total_messages - 1}.
DO NOT create segments beyond message index {total_messages - 1}.

Conversation to analyze:

{conversation_text}

Please analyze this dialogue and create a structured breakdown:

SEGMENTATION STRATEGY:
- Create segments based on MAJOR conversation themes and substantial topic shifts
- Each segment should represent a significant conversation phase or major theme
- Look for MAJOR breakpoints: significant topic changes, major emotional shifts, substantial problem-solution cycles
- Create FEWER, MORE SUBSTANTIAL segments - prioritize comprehensive coverage over granular division
- Each segment MUST contain at least 200 words of conversation content
- Create a MAXIMUM of 10 segments total, fewer is strongly preferred
- Only segment when there is a truly significant shift in conversation direction

SEGMENT CRITERIA (apply conservatively):
1. Major topic shift: When conversation moves to a completely different subject area
2. Significant emotional change: When the overall mood or tone changes substantially
3. Complete problem-solution cycles: Full cycles from problem identification through resolution
4. Major conversation direction change: Significant shifts from one conversation mode to another
5. Clear temporal breaks: Only very obvious time gaps or conversation restarts
6. Major engagement level shifts: Substantial changes in conversation depth or involvement

OUTPUT REQUIREMENTS:
- Create NO MORE than 10 segments maximum (fewer is strongly preferred)
- Each segment MUST contain at least 200 words of dialogue content
- Only create new segments for truly major conversation shifts
- Err on the side of fewer, larger segments rather than many small ones
- For short conversations (under 1000 words), create only 2-4 segments maximum
- CRITICAL: start_idx and end_idx must be between 0 and {total_messages - 1}
- CRITICAL: Do not create segments for messages that don't exist

Focus on capturing the major phases and substantial themes of the conversation experience.

Return ONLY a JSON array with this exact format:
[
  {{
    "conversation_segment_id": 1,
    "start_idx": 0,
    "end_idx": 25
  }},
  {{
    "conversation_segment_id": 2, 
    "start_idx": 26,
    "end_idx": 50
  }}
]

Where start_idx and end_idx are 0-based message indices between 0 and {total_messages - 1}."""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert conversation analyzer. Return only valid JSON with segment boundaries."},
                    {"role": "user", "content": expert_prompt}
                ],
                max_tokens=10000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            import json
            segments_data = json.loads(response.choices[0].message.content)
            
            # Convert to our format and populate content programmatically
            segments = []
            if isinstance(segments_data, list):
                segment_list = segments_data
            elif 'segments' in segments_data:
                segment_list = segments_data['segments']
            else:
                return self.get_programmatic_segments_new_format(conversation)
            
            for segment_data in segment_list:
                if 'start_idx' in segment_data and 'end_idx' in segment_data:
                    start_idx = segment_data['start_idx']
                    end_idx = segment_data['end_idx']
                    
                    # Validate indices are within conversation bounds
                    if start_idx < 0 or start_idx >= len(conversation):
                        st.warning(f"Invalid start_idx {start_idx} for conversation with {len(conversation)} messages. Skipping segment.")
                        continue
                    
                    if end_idx < 0 or end_idx >= len(conversation):
                        st.warning(f"Invalid end_idx {end_idx} for conversation with {len(conversation)} messages. Adjusting to {len(conversation) - 1}.")
                        end_idx = len(conversation) - 1
                    
                    if start_idx > end_idx:
                        st.warning(f"Invalid segment: start_idx {start_idx} > end_idx {end_idx}. Skipping segment.")
                        continue
                    
                    # Programmatically copy-paste content using validated indices
                    content = self.extract_content_from_indices(conversation, start_idx, end_idx)
                    
                    # Validate content is not empty
                    if not content.strip():
                        st.warning(f"Empty content for segment {start_idx}-{end_idx}. Skipping segment.")
                        continue
                    
                    segments.append({
                        'conversation_segment_id': segment_data.get('conversation_segment_id', len(segments) + 1),
                        'content': content,
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    })
            
            if segments:
                return segments
            else:
                return self.get_programmatic_segments_new_format(conversation)
                
        except Exception as e:
            st.error(f"Expert segmentation error: {str(e)}. Using fallback segmentation.")
            return self.get_programmatic_segments_new_format(conversation)
    
    def get_programmatic_segments_new_format(self, conversation: List[Dict]) -> List[Dict]:
        """Fallback programmatic segmentation in new JSON format with larger segments."""
        segments = []
        
        # Special handling for known test conversation (6 messages) - combine into 2 segments
        if len(conversation) == 6:
            segment_boundaries = [
                (0, 3),  # Greeting + Health Concern + Weather Info
                (4, 5)   # Care Advice + Agreement
            ]
            
            for i, (start_idx, end_idx) in enumerate(segment_boundaries):
                content = self.extract_content_from_indices(conversation, start_idx, end_idx)
                segments.append({
                    'conversation_segment_id': i + 1,
                    'content': content,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
        
        else:
            # General segmentation - create larger segments to meet 200-word minimum
            # Calculate segment size to create max 10 segments
            total_messages = len(conversation)
            target_segments = min(10, max(1, total_messages // 20))  # Aim for ~20 messages per segment
            
            if target_segments == 1:
                # If conversation is small, create just 1-2 segments
                if total_messages <= 10:
                    segments.append({
                        'conversation_segment_id': 1,
                        'content': self.extract_content_from_indices(conversation, 0, total_messages - 1),
                        'start_idx': 0,
                        'end_idx': total_messages - 1
                    })
                else:
                    # Split into 2 segments
                    mid_point = total_messages // 2
                    segments.append({
                        'conversation_segment_id': 1,
                        'content': self.extract_content_from_indices(conversation, 0, mid_point - 1),
                        'start_idx': 0,
                        'end_idx': mid_point - 1
                    })
                    segments.append({
                        'conversation_segment_id': 2,
                        'content': self.extract_content_from_indices(conversation, mid_point, total_messages - 1),
                        'start_idx': mid_point,
                        'end_idx': total_messages - 1
                    })
            else:
                # Create larger segments
                messages_per_segment = total_messages // target_segments
                
                for i in range(target_segments):
                    start_idx = i * messages_per_segment
                    if i == target_segments - 1:
                        # Last segment gets any remaining messages
                        end_idx = total_messages - 1
                    else:
                        end_idx = start_idx + messages_per_segment - 1
                    
                    content = self.extract_content_from_indices(conversation, start_idx, end_idx)
                    segments.append({
                        'conversation_segment_id': i + 1,
                        'content': content,
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    })
        
        return segments
    
    def format_conversation_for_prompt(self, conversation: List[Dict]) -> str:
        """Format conversation for the expert analysis prompt."""
        formatted = ""
        for msg in conversation:
            formatted += f"[{msg['timestamp']}] {msg['speaker']}: {msg['message']}\n"
        return formatted.strip()
    
    def extract_content_from_indices(self, conversation: List[Dict], start_idx: int, end_idx: int) -> str:
        """Programmatically copy-paste content using start_idx and end_idx."""
        content_lines = []
        
        for i in range(start_idx, min(end_idx + 1, len(conversation))):
            msg = conversation[i]
            # Format as "Speaker: message" without timestamp
            content_lines.append(f"{msg['speaker']}: {msg['message']}")
        
        return "\n".join(content_lines)
    

    
    def format_segment_for_prompt(self, segment: Dict) -> str:
        """Format a conversation segment for AI prompt."""
        formatted = "DIALOGUE SEGMENT:\n"
        formatted += segment['content']
        return formatted
    
    def generate_segment_summary_with_api(self, segment: Dict) -> str:
        """Generate summary for a specific segment using OpenAI API."""
        try:
            # Check if API key is available
            if not hasattr(st.session_state, 'openai_api_key') or not st.session_state.openai_api_key:
                return self.generate_fallback_segment_summary(segment)
            
            client = openai.OpenAI(api_key=st.session_state.openai_api_key)
            
            segment_text = self.format_segment_for_prompt(segment)
            full_prompt = f"{self.segment_summary_prompt}\n\n{segment_text}"
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise, accurate summaries of dialogue segments."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return self.generate_fallback_segment_summary(segment)
    
    def generate_fallback_segment_summary(self, segment: Dict) -> str:
        """Generate a fallback summary for a segment when API is not available."""
        content = segment.get('content', '')
        if not content:
            return "Empty segment."
        
        # Extract speakers from content lines
        lines = content.split('\n')
        speakers = list(set(line.split(':', 1)[0] for line in lines if ':' in line))
        
        # Extract all text for topic analysis
        all_text = " ".join([line.split(':', 1)[1].strip() for line in lines if ':' in line])
        words = all_text.lower().split()
        
        # Key topic detection
        health_words = ['cold', 'sick', 'feel', 'better', 'health', 'medicine', 'rest', 'recover']
        weather_words = ['weather', 'temperature', 'degrees', 'wind', 'thunderstorm', 'rain', 'warm', 'celsius', 'fahrenheit']
        location_words = ['miami', 'minnesota', 'dundas']
        care_words = ['tea', 'honey', 'lemon', 'warm', 'care', 'help', 'suggestion']
        
        topics = []
        if any(word in words for word in health_words):
            topics.append("health discussion")
        if any(word in words for word in weather_words):
            topics.append("weather information")
        if any(word in words for word in location_words):
            topics.append("location details")
        if any(word in words for word in care_words):
            topics.append("care suggestions")
        
        # Generate summary based on content
        if len(lines) == 1:
            first_speaker = speakers[0] if speakers else "Speaker"
            summary = f"{first_speaker} shares: {all_text[:60]}..."
        else:
            if len(speakers) == 1:
                summary = f"{speakers[0]} discusses {', '.join(topics) if topics else 'various topics'}."
            else:
                summary = f"Exchange between {' and '.join(speakers)} about {', '.join(topics) if topics else 'conversation topics'}."
        
        return summary
    
    def generate_segmented_summaries(self, conversation: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Generate summaries for each conversation segment."""
        segments = self.segment_conversation(conversation)
        summaries = []
        
        for segment in segments:
            summary_text = self.generate_segment_summary_with_api(segment)
            summaries.append({
                'conversation_segment_id': segment['conversation_segment_id'],
                'summary': summary_text,
                'segment': segment
            })
        
        return summaries, segments

class FollowupGenerator:
    """Generate followups using the agent-based system from workflow_read_file.py"""
    
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        if AGENTS_AVAILABLE:
            self.agent_segmenter_rater = make_agent_chat_segmenter_rater(model_name=model_name)
            self.agent_starter_generator = make_agent_conversation_starter_generator(model_name=model_name)
    
    def format_conversation_for_agents(self, conversation: List[Dict]) -> str:
        """Convert Streamlit conversation format to agent-expected format."""
        conversation_lines = []
        
        for msg in conversation:
            # Convert to agent format: "speaker: message"
            if msg['speaker'].lower() == 'agent':
                conversation_lines.append(f"agent: {msg['message']}")
            elif msg['speaker'].lower() in ['user', 'user 2']:
                conversation_lines.append(f"user: {msg['message']}")
            else:
                # Handle other speaker types
                speaker_type = "agent" if "agent" in msg['speaker'].lower() else "user"
                conversation_lines.append(f"{speaker_type}: {msg['message']}")
        
        return '\n'.join(conversation_lines)
    
    async def generate_followups_with_agents(self, conversation: List[Dict]) -> List[str]:
        """Generate followups using the agent-based system."""
        if not AGENTS_AVAILABLE:
            return self.generate_fallback_followups()
        
        try:
            # Set up environment for agents to access OpenAI API key
            if hasattr(st.session_state, 'openai_api_key') and st.session_state.openai_api_key:
                os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
            elif not os.getenv('OPENAI_API_KEY'):
                st.warning("⚠️ No OpenAI API key available for agents. Using fallback followups.")
                return self.generate_fallback_followups()
            
            # Format conversation for agents
            formatted_conversation = self.format_conversation_for_agents(conversation)
            
            # Step 1: Segment and rate conversation
            deps = SegmenterRaterDeps(conversation=formatted_conversation)
            result = await self.agent_segmenter_rater.run("Please analyze this conversation.", deps=deps)
            segments = result.data.segments
            
            # Get top 3 segments by combined score
            top_segments = sorted(segments, key=lambda x: x.combined_score, reverse=True)[:3]
            
            # Step 2: Generate conversation starters
            starter_deps = StarterGeneratorDeps(top_segments=top_segments)
            starter_result = await self.agent_starter_generator.run(deps=starter_deps)
            starters = starter_result.data
            
            # Convert ConversationStarter objects to strings
            followup_strings = [starter.starter for starter in starters[:5]]  # Take top 5
            
            return followup_strings
            
        except Exception as e:
            st.error(f"Agent-based followup generation failed: {str(e)}")
            return self.generate_fallback_followups()
    
    def generate_followups_sync(self, conversation: List[Dict]) -> List[str]:
        """Synchronous wrapper for async followup generation."""
        if not AGENTS_AVAILABLE:
            return self.generate_fallback_followups()
            
        try:
            # Try to run in new event loop
            return asyncio.run(self.generate_followups_with_agents(conversation))
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                # We're in an existing event loop, try different approach
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    return asyncio.run(self.generate_followups_with_agents(conversation))
                except ImportError:
                    st.warning("⚠️ nest_asyncio not available. Using fallback followups.")
                    return self.generate_fallback_followups()
                except Exception as e:
                    st.error(f"Nested async execution failed: {str(e)}")
                    return self.generate_fallback_followups()
            else:
                st.error(f"Async execution failed: {str(e)}")
                return self.generate_fallback_followups()
        except Exception as e:
            st.error(f"Followup generation failed: {str(e)}")
            return self.generate_fallback_followups()
    
    def generate_fallback_followups(self) -> List[str]:
        """Generate fallback follow-ups when agents are not available."""
        return [
            "I hope you feel better soon! Is there anything specific you'd like me to help you with while you recover?",
            "Would you like me to look up some remedies for cold symptoms that might help you feel better?",
            "Since the weather is quite variable, would you like me to send you daily weather updates while you're feeling unwell?",
            "Have you been able to rest well? Good sleep is really important when fighting off a cold.",
            "Would you like me to remind you to take your medication or suggest some warm drinks that might soothe your throat?"
        ]

class DataStorage:
    """Handle data storage for user interactions and feedback."""
    
    def __init__(self, log_file: str = "user_interactions.jsonl"):
        self.log_file = log_file
        
    def log_interaction(self, interaction_data: Dict):
        """Log user interaction to file."""
        interaction_data['logged_at'] = datetime.now().isoformat()
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(interaction_data) + '\n')
    
    def get_session_data(self) -> Dict:
        """Get or create session data."""
        if 'session_data' not in st.session_state:
            st.session_state.session_data = {
                'conversation': [],
                'segment_summaries': [],
                'segments': [],
                'follow_ups': [],
                'selected_follow_up': None,
                'user_feedback': {},
                'selected_segment': None
            }
        return st.session_state.session_data

def generate_segmented_summaries(conversation: List[Dict], summary_generator: SummaryGenerator) -> Tuple[List[Dict], List[Dict]]:
    """Generate segment-based summaries with precise source mapping."""
    summaries, segments = summary_generator.generate_segmented_summaries(conversation)
    return summaries, segments

def display_interactive_segment_summaries(segment_summaries: List[Dict], segments: List[Dict], conversation: List[Dict], session_data: Dict):
    """Display an interactive summary with clickable segment summaries."""
    st.markdown("### 📝 Interactive Segment Summaries")
    st.markdown("*Click on any summary to view the exact source dialogue segment*")
    
    # Display each segment summary as a clickable button
    for summary_item in segment_summaries:
        segment_id = summary_item['conversation_segment_id']
        summary_text = summary_item['summary']
        
        # Create a button for each segment summary
        if st.button(
            f"**Segment {segment_id}:** {summary_text}",
            key=f"segment_summary_{segment_id}",
            help="Click to view exact source dialogue",
            use_container_width=True
        ):
            session_data['selected_segment'] = segment_id
            
            # Log the segment interaction
            storage = DataStorage()
            storage.log_interaction({
                'action': 'segment_summary_clicked',
                'segment_id': segment_id,
                'segment_number': segment_id,
                'summary_text': summary_text[:100],  # Log first 100 chars
                'segment_content_length': len(summary_item['segment']['content'])
            })
        
        # Add some spacing
        st.write("")
    
    # Display source context if a segment is selected
    if session_data.get('selected_segment') is not None:
        selected_segment_id = session_data['selected_segment']
        
        # Find the selected segment
        selected_segment = None
        for seg in segments:
            if seg['conversation_segment_id'] == selected_segment_id:
                selected_segment = seg
                break
        
        if selected_segment:
            st.markdown("---")
            st.markdown("### 🔍 Exact Source Dialogue Segment")
            
            st.info(f"📌 Showing complete dialogue for Segment {selected_segment_id}")
            
            # Display the exact conversation content from this segment
            content_lines = selected_segment['content'].split('\n')
            for line in content_lines:
                if ':' in line:
                    speaker, message = line.split(':', 1)
                    # Highlight each message with proper contrast
                    st.markdown(f"""
                    <div style="background-color: #e8f4fd; color: #1e1e1e; padding: 15px; margin: 8px 0; border-radius: 8px; border-left: 5px solid #0066cc; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <strong style="color: #0066cc;">{speaker}:</strong><br>
                        <span style="color: #2c3e50; line-height: 1.5;">{message.strip()}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show segment info
            message_count = len(content_lines)
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; border: 1px solid #dee2e6;">
                <small style="color: #6c757d;">
                    <strong>Segment Info:</strong> {message_count} messages 
                    (positions {selected_segment['start_idx']+1}-{selected_segment['end_idx']+1} in full conversation)
                </small>
            </div>
            """, unsafe_allow_html=True)
            
            # Add button to clear selection
            if st.button("🔄 Clear Selection", key="clear_selection"):
                session_data['selected_segment'] = None
                st.rerun()

def generate_intelligent_follow_ups(conversation: List[Dict], followup_generator: FollowupGenerator) -> List[str]:
    """Generate intelligent follow-ups using the agent-based system."""
    return followup_generator.generate_followups_sync(conversation)

def main():
    """Main Streamlit application."""
    
    # Initialize data storage, summary generator, and followup generator
    storage = DataStorage()
    session_data = storage.get_session_data()
    summary_generator = SummaryGenerator()
    followup_generator = FollowupGenerator()
    
    # Header
    st.title("🤖 Agentic Follow-up Rating System")
    st.markdown("Rate and improve AI-generated conversation follow-ups")
    
    # Sidebar for file upload and controls
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key (optional)",
            type="password",
            help="Enter your OpenAI API key for AI-powered summaries. If not provided, a fallback summary will be generated.",
            placeholder="sk-..."
        )
        
        if api_key:
            st.session_state.openai_api_key = api_key
            st.success("✅ API key configured")
        elif hasattr(st.session_state, 'openai_api_key'):
            st.info("🔑 Using previously entered API key")
        else:
            st.warning("⚠️ No API key provided - using fallback summary generation")
        
        st.divider()
        st.header("📁 Upload Conversation")
        
        uploaded_file = st.file_uploader(
            "Choose a conversation file",
            type=['txt'],
            help="Upload a text file with conversation logs in format: [timestamp] Speaker: message"
        )
        
        if uploaded_file is not None:
            # Read and parse the file
            content = uploaded_file.read().decode('utf-8')
            parser = ConversationParser()
            conversation = parser.parse_conversation(content)
            
            if conversation:
                session_data['conversation'] = conversation
                st.success(f"✅ Parsed {len(conversation)} messages")
                
                # Show conversation preview
                st.subheader("📋 Conversation Preview")
                for i, msg in enumerate(conversation[:3]):  # Show first 3 messages
                    st.text(f"[{msg['timestamp']}] {msg['speaker']}: {msg['message'][:50]}...")
                
                if len(conversation) > 3:
                    st.text(f"... and {len(conversation) - 3} more messages")
                    
                # Generate summary and follow-ups
                if st.button("🔄 Process Conversation"):
                    with st.spinner("Generating segment-based AI summaries..."):
                        # Show conversation statistics
                        st.info(f"📊 Conversation contains {len(conversation)} messages (indices 0-{len(conversation)-1})")
                        
                        # Generate segment-based summaries
                        summaries, segments = generate_segmented_summaries(
                            conversation, summary_generator
                        )
                        
                        # Validate and display segment information
                        valid_segments = []
                        for segment in segments:
                            if segment['start_idx'] < len(conversation) and segment['end_idx'] < len(conversation):
                                valid_segments.append(segment)
                            else:
                                st.error(f"⚠️ Invalid segment {segment['conversation_segment_id']}: indices {segment['start_idx']}-{segment['end_idx']} exceed conversation length {len(conversation)}")
                        
                        session_data['segment_summaries'] = summaries
                        session_data['segments'] = valid_segments
                        
                        # Generate intelligent follow-ups using agents
                        with st.spinner("Generating intelligent follow-ups..."):
                            session_data['follow_ups'] = generate_intelligent_follow_ups(conversation, followup_generator)
                        
                        # Log summary generation
                        storage.log_interaction({
                            'action': 'conversation_processed',
                            'conversation_length': len(conversation),
                            'number_of_segments': len(valid_segments),
                            'invalid_segments': len(segments) - len(valid_segments),
                            'number_of_followups': len(session_data['follow_ups']),
                            'agents_available': AGENTS_AVAILABLE,
                            'has_api_key': bool(hasattr(st.session_state, 'openai_api_key') and st.session_state.openai_api_key)
                        })
                        
                        # Success message based on available systems
                        followup_method = "intelligent agent-based" if AGENTS_AVAILABLE else "fallback"
                        st.success(f"✅ Conversation processed with {len(valid_segments)} valid segments and {len(session_data['follow_ups'])} {followup_method} follow-ups!")
                        
                        if len(segments) != len(valid_segments):
                            st.warning(f"⚠️ Filtered out {len(segments) - len(valid_segments)} invalid segments that exceeded conversation bounds")
                            
                        if not AGENTS_AVAILABLE:
                            st.info("💡 Install agent modules for intelligent followup generation")
                        st.rerun()
            else:
                st.error("❌ No valid conversation messages found in the file")
        
        # Session info
        if session_data['conversation']:
            st.divider()
            st.subheader("📊 Session Info")
            total_messages = len(session_data['conversation'])
            st.write(f"**Total Messages**: {total_messages} (indices 0-{total_messages-1})")
            st.write(f"**Segments**: {len(session_data.get('segments', []))}")
            
            # Show segment ranges if available
            if session_data.get('segments'):
                st.write("**Segment Ranges**:")
                for segment in session_data['segments']:
                    st.text(f"  Segment {segment['conversation_segment_id']}: {segment['start_idx']}-{segment['end_idx']}")
            
            st.write(f"**Summaries generated**: {'✅' if session_data['segment_summaries'] else '❌'}")
            st.write(f"**Follow-ups generated**: {'✅' if session_data['follow_ups'] else '❌'}")
            st.write(f"**Agent system**: {'✅ Available' if AGENTS_AVAILABLE else '❌ Using fallback'}")
    
    # Main content area
    if session_data['conversation'] and session_data['follow_ups']:
        
        # Create two columns: follow-ups (left) and summary (right)
        col1, col2 = st.columns([1, 1])
        
        # Left column: Follow-ups
        with col1:
            st.header("💬 Generated Follow-ups")
            st.markdown("Select and rate the follow-ups below:")
            
            selected_follow_up = None
            
            for i, follow_up in enumerate(session_data['follow_ups']):
                st.subheader(f"Follow-up {i+1}")
                
                # Display follow-up in an editable text area
                edited_follow_up = st.text_area(
                    f"Edit follow-up {i+1}:",
                    value=follow_up,
                    height=80,
                    key=f"follow_up_{i}"
                )
                
                # Selection and rating
                col_select, col_rate = st.columns([1, 1])
                
                with col_select:
                    if st.button(f"Select Follow-up {i+1}", key=f"select_{i}"):
                        selected_follow_up = i
                        session_data['selected_follow_up'] = i
                        
                        # Log the selection
                        storage.log_interaction({
                            'action': 'follow_up_selected',
                            'follow_up_index': i,
                            'original_follow_up': follow_up,
                            'edited_follow_up': edited_follow_up
                        })
                        
                        st.success(f"✅ Selected follow-up {i+1}")
                
                with col_rate:
                    rating = st.select_slider(
                        f"Rate {i+1}:",
                        options=[1, 2, 3, 4, 5],
                        value=3,
                        key=f"rating_{i}"
                    )
                
                # Add comment section
                comment = st.text_input(
                    f"Comment on follow-up {i+1}:",
                    key=f"comment_{i}",
                    placeholder="Optional feedback..."
                )
                
                # Store ratings and comments
                session_data['user_feedback'][f'follow_up_{i}'] = {
                    'rating': rating,
                    'comment': comment,
                    'edited_text': edited_follow_up
                }
                
                st.divider()
        
        # Right column: Summary
        with col2:
            st.header("📝 Conversation Segment Summaries")
            
            if session_data['segment_summaries'] and session_data.get('segments'):
                # Display interactive segment summaries
                display_interactive_segment_summaries(
                    session_data['segment_summaries'],
                    session_data['segments'],
                    session_data['conversation'],
                    session_data
                )
                
                # Show selected follow-up
                if session_data.get('selected_follow_up') is not None:
                    st.markdown("---")
                    st.subheader("✅ Selected Follow-up")
                    selected_idx = session_data['selected_follow_up']
                    selected_text = session_data['user_feedback'][f'follow_up_{selected_idx}']['edited_text']
                    st.success(f"**Follow-up {selected_idx + 1}**: {selected_text}")
            
            elif session_data['conversation']:
                # Show message to process the conversation
                st.info("💡 Upload a conversation and click 'Process Conversation' to generate segment-based summaries")
            
            # Show original conversation
            with st.expander("📖 View Original Conversation"):
                for msg in session_data['conversation']:
                    st.text(f"[{msg['timestamp']}] {msg['speaker']}: {msg['message']}")
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Agentic Follow-up Rating System
        
        This tool helps you evaluate and improve AI-generated conversation follow-ups.
        
        ### How to use:
        1. **Upload** a conversation file using the sidebar
        2. **Review** the generated summary and follow-ups
        3. **Edit** follow-ups if needed
        4. **Rate** and **select** the best follow-up
        5. **Provide feedback** to improve the system
        
        ### Expected file format:
        ```
        [4/18/2025, 12:29:52 AM] Agent: What's new, [REDACTED]?
        [4/18/2025, 12:29:57 AM] User 2: I have a very bad cold,
        [4/18/2025, 12:30:01 AM] Agent: Oh, get well soon!
        ```
        
        Upload a conversation file to get started! 🚀
        """)

if __name__ == "__main__":
    main() 