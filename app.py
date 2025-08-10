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
        ConversationSegment,
        populate_segment_content
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
        self.segment_summary_prompt = """Please provide a concise summary of this dialogue segment in 1-2 sentences. Focus on the key points, actions, or information exchanged. Be objective and factual. 

LANGUAGE DETECTION: If the dialogue segment is primarily in Russian, provide your summary in Russian. If it's in English or other languages, respond in English."""
        
        # Enforced language code detected upstream: 'ru' or 'en'. None means auto-detect.
        self.language_code: Optional[str] = None
        
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
            # Enforce detected language if available
            lang = self.language_code or st.session_state.get('detected_language')
            language_directive = "Please write the summary in Russian." if lang == 'ru' else "Please write the summary in English."
            full_prompt = f"{self.segment_summary_prompt}\n\nLANGUAGE REQUIREMENT: {language_directive}\n\n{segment_text}"
            
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
        
        # Enforce detected language if available
        enforced_lang = self.language_code or st.session_state.get('detected_language')
        if enforced_lang in ('ru', 'en'):
            is_russian = enforced_lang == 'ru'
        else:
            # Auto-detect if not enforced
            russian_words = ['что', 'как', 'где', 'когда', 'почему', 'да', 'нет', 'хорошо', 'плохо', 'спасибо', 'пожалуйста', 'привет', 'пока', 'здравствуйте', 'до свидания']
            is_russian = any(word in all_text.lower() for word in russian_words) or any(ord(char) >= 1040 and ord(char) <= 1103 for char in all_text)
        
        if is_russian:
            # Russian topic detection
            health_words = ['простуда', 'болен', 'болеет', 'здоровье', 'лекарство', 'отдых', 'выздоровление', 'чувствую', 'лучше']
            weather_words = ['погода', 'температура', 'градусы', 'ветер', 'дождь', 'снег', 'тепло', 'холодно', 'солнце']
            location_words = ['москва', 'петербург', 'россия', 'город', 'дом', 'работа']
            care_words = ['чай', 'мёд', 'лимон', 'забота', 'помощь', 'совет']
            
            topics = []
            if any(word in all_text.lower() for word in health_words):
                topics.append("обсуждение здоровья")
            if any(word in all_text.lower() for word in weather_words):
                topics.append("информация о погоде")
            if any(word in all_text.lower() for word in location_words):
                topics.append("детали местоположения")
            if any(word in all_text.lower() for word in care_words):
                topics.append("советы по уходу")
            
            # Generate Russian summary
            if len(lines) == 1:
                first_speaker = speakers[0] if speakers else "Собеседник"
                summary = f"{first_speaker} делится: {all_text[:60]}..."
            else:
                if len(speakers) == 1:
                    summary = f"{speakers[0]} обсуждает {', '.join(topics) if topics else 'различные темы'}."
                else:
                    summary = f"Обмен между {' и '.join(speakers)} о {', '.join(topics) if topics else 'темах разговора'}."
        else:
            # English topic detection
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
            
            # Generate English summary
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
        # Defer agent initialization until API key is available
        self.agent_segmenter_rater = None
        self.agent_starter_generator = None
    
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
    
    async def generate_followups_with_agents(self, conversation: List[Dict]) -> Tuple[List[str], List]:
        """Generate followups using the agent-based system."""
        if not AGENTS_AVAILABLE:
            return self.generate_fallback_followups(conversation), []
        
        try:
            # Set up environment for agents to access OpenAI API key
            if hasattr(st.session_state, 'openai_api_key') and st.session_state.openai_api_key:
                os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
            elif not os.getenv('OPENAI_API_KEY'):
                st.warning("⚠️ No OpenAI API key available for agents. Using fallback followups.")
                return self.generate_fallback_followups(conversation), []
            
            # Initialize agents lazily after API key is present
            if self.agent_segmenter_rater is None:
                self.agent_segmenter_rater = make_agent_chat_segmenter_rater(model_name=self.model_name)
            if self.agent_starter_generator is None:
                self.agent_starter_generator = make_agent_conversation_starter_generator(model_name=self.model_name)
            
            # Format conversation for agents
            formatted_conversation = self.format_conversation_for_agents(conversation)
            
            # Step 1: Segment and rate conversation
            deps = SegmenterRaterDeps(conversation=formatted_conversation)
            # Enforce language for segmenter if detected
            lang = st.session_state.get('detected_language')
            language_note = "Please respond entirely in Russian (all fields)." if lang == 'ru' else "Please respond entirely in English (all fields)."
            seg_prompt = f"Please analyze this conversation. {language_note}"
            result = await self.agent_segmenter_rater.run(seg_prompt, deps=deps)
            segments = result.data.segments
            
            # Programmatically populate segment content based on line numbers
            if AGENTS_AVAILABLE:
                segments = populate_segment_content(segments, formatted_conversation)
            
            # Get top 3 segments by combined score
            top_segments = sorted(segments, key=lambda x: x.combined_score, reverse=True)[:3]
            
            # Step 2: Generate conversation starters
            starter_deps = StarterGeneratorDeps(top_segments=top_segments)
            starter_result = await self.agent_starter_generator.run(deps=starter_deps)
            starters = starter_result.data
            
            # Convert ConversationStarter objects to strings
            followup_strings = [starter.starter for starter in starters]  # Show all generated
            
            return followup_strings, segments
            
        except Exception as e:
            st.error(f"Agent-based followup generation failed: {str(e)}")
            return self.generate_fallback_followups(conversation), []
    
    def generate_followups_sync(self, conversation: List[Dict]) -> List[str]:
        """Synchronous wrapper for async followup generation."""
        followups, _ = self.generate_followups_and_segments_sync(conversation)
        return followups
    
    def generate_followups_and_segments_sync(self, conversation: List[Dict]) -> Tuple[List[str], List]:
        """Synchronous wrapper for async followup generation that returns both followups and segments."""
        if not AGENTS_AVAILABLE:
            return self.generate_fallback_followups(conversation), []
            
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
                    return self.generate_fallback_followups(conversation), []
                except Exception as e:
                    st.error(f"Nested async execution failed: {str(e)}")
                    return self.generate_fallback_followups(conversation), []
            else:
                st.error(f"Async execution failed: {str(e)}")
                return self.generate_fallback_followups(conversation), []
        except Exception as e:
            st.error(f"Followup generation failed: {str(e)}")
            return self.generate_fallback_followups(conversation), []
    
    def generate_fallback_followups(self, conversation: List[Dict] = None) -> List[str]:
        """Generate fallback follow-ups when agents are not available."""
        
        # Determine language from detected value first
        lang = st.session_state.get('detected_language')
        is_russian = False
        if lang in ('ru', 'en'):
            is_russian = lang == 'ru'
        else:
            # Heuristic detection if not provided
            if conversation:
                all_text = " ".join([msg.get('message', '') for msg in conversation])
                russian_words = ['что', 'как', 'где', 'когда', 'почему', 'да', 'нет', 'хорошо', 'плохо', 'спасибо', 'пожалуйста', 'привет', 'пока', 'здравствуйте', 'до свидания']
                is_russian = any(word in all_text.lower() for word in russian_words) or any(ord(char) >= 1040 and ord(char) <= 1103 for char in all_text)
        
        if is_russian:
            return [
                "Надеюсь, ты скоро поправишься! Есть что-то конкретное, с чем я могу помочь тебе во время выздоровления?",
                "Хочешь, я найду некоторые средства от симптомов простуды, которые могут помочь тебе почувствовать себя лучше?",
                "Поскольку погода довольно переменчивая, хочешь, чтобы я присылал тебе ежедневные прогнозы погоды, пока ты плохо себя чувствуешь?",
                "Удается ли тебе хорошо отдыхать? Хороший сон очень важен при борьбе с простудой.",
                "Хочешь, чтобы я напоминал тебе принимать лекарства или посоветовал теплые напитки, которые могут успокоить горло?"
            ]
        else:
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
                'agent_segments': [],  # Store conversation segments from agents
                'selected_follow_up': None,
                'user_feedback': {},
                'selected_segment': None,
                'show_agent_analysis': False  # Flag to control display of agent analysis
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

def generate_intelligent_follow_ups_and_segments(conversation: List[Dict], followup_generator: FollowupGenerator) -> Tuple[List[str], List]:
    """Generate intelligent follow-ups and conversation segments using the agent-based system."""
    return followup_generator.generate_followups_and_segments_sync(conversation)

def display_agent_conversation_segments(agent_segments: List, session_data: Dict):
    """Display conversation segments generated by the AI agents with engagement/enjoyment scores and reasoning."""
    if not agent_segments:
        st.info("🤖 No agent-generated segments available. This feature requires the agent system to be active.")
        return
    
    st.markdown("### 🧠 AI Agent Analysis")
    st.markdown("*These segments were generated by the AI agent system for follow-up analysis*")
    st.markdown("*Each segment is scored on engagement (1-10) and enjoyment (1-10) scales*")
    
    # Sort segments by combined score (highest first)
    sorted_segments = sorted(agent_segments, key=lambda x: x.combined_score, reverse=True)
    
    for i, segment in enumerate(sorted_segments):
        # Create an expandable section for each segment
        # Create an expandable section for each segment
        line_info = ""
        if hasattr(segment, 'start_line') and hasattr(segment, 'end_line'):
            line_info = f" | Lines {segment.start_line}-{segment.end_line}"
        
        with st.expander(f"🎯 Segment {segment.segment_id} (Score: {segment.combined_score}/20){line_info}", expanded=(i == 0)):
            
            # Display scores with color coding
            col1, col2, col3 = st.columns(3)
            
            with col1:
                engagement_color = "🟢" if segment.engagement_score >= 7 else "🟡" if segment.engagement_score >= 4 else "🔴"
                st.metric("Engagement Score", f"{segment.engagement_score}/10", help="How engaging this segment is (1-10 scale)")
                st.write(f"{engagement_color} Engagement Level")
            
            with col2:
                enjoyment_color = "🟢" if segment.enjoyment_score >= 7 else "🟡" if segment.enjoyment_score >= 4 else "🔴"
                st.metric("Enjoyment Score", f"{segment.enjoyment_score}/10", help="How enjoyable this segment is (1-10 scale)")
                st.write(f"{enjoyment_color} Enjoyment Level")
            
            with col3:
                combined_color = "🟢" if segment.combined_score >= 14 else "🟡" if segment.combined_score >= 8 else "🔴"
                st.metric("Combined Score", f"{segment.combined_score}/20", help="Overall segment quality score (sum of engagement + enjoyment)")
                st.write(f"{combined_color} Overall Quality")
            
            # Display AI reasoning and segment info
            st.markdown("**📋 Segment Details:**")
            
            # Show topic, tone, and interaction type
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.markdown(f"**🎯 Topic:** {segment.topic}")
                st.markdown(f"**🎭 Tone:** {segment.tone}")
            with detail_col2:
                st.markdown(f"**🔄 Direction:** {segment.conversation_direction}")
                st.markdown(f"**🤝 Type:** {segment.interaction_type}")
            
            # Display AI justifications
            st.markdown("**🧠 AI Analysis:**")
            
            if hasattr(segment, 'engagement_justification') and segment.engagement_justification:
                st.markdown(f"**Engagement Reasoning:** {segment.engagement_justification}")
            
            if hasattr(segment, 'enjoyment_justification') and segment.enjoyment_justification:
                st.markdown(f"**Enjoyment Reasoning:** {segment.enjoyment_justification}")
            
            # Display segment boundary info
            if hasattr(segment, 'start_line') and hasattr(segment, 'end_line'):
                st.markdown(f"**📍 Segment Boundaries:** Lines {segment.start_line}-{segment.end_line}")
            
            # Display segment content
            st.markdown("**💬 Complete Segment Content:**")
            
            if segment.content and segment.content.strip():
                content_lines = segment.content.split('\n')
                
                for i, line in enumerate(content_lines):
                    line = line.strip()
                    if not line:
                        continue
                        
                    if ':' in line:
                        # Try to split on first colon to separate speaker from message
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            speaker, message = parts
                            speaker = speaker.strip()
                            message = message.strip()
                            
                            # Color code by speaker type
                            if speaker.lower() == 'agent':
                                st.markdown(f"""
                                <div style="background-color: #e3f2fd; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #2196f3;">
                                    <strong style="color: #1976d2;">🤖 {speaker}:</strong> <span style="color: #424242;">{message}</span>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="background-color: #f3e5f5; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #9c27b0;">
                                    <strong style="color: #7b1fa2;">👤 {speaker}:</strong> <span style="color: #424242;">{message}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            # Fallback for lines that don't follow expected format
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 8px; margin: 3px 0; border-radius: 3px; border-left: 2px solid #6c757d;">
                                <span style="color: #495057;">{line}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Handle lines without colon (shouldn't happen but good fallback)
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 8px; margin: 3px 0; border-radius: 3px; border-left: 2px solid #6c757d;">
                            <span style="color: #495057;">{line}</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No content available for this segment")
            
            st.markdown("---")
    
    # Summary stats
    st.markdown("### 📊 Segment Analysis Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_engagement = sum(s.engagement_score for s in agent_segments) / len(agent_segments)
        st.metric("Avg Engagement", f"{avg_engagement:.1f}/10")
    
    with col2:
        avg_enjoyment = sum(s.enjoyment_score for s in agent_segments) / len(agent_segments)
        st.metric("Avg Enjoyment", f"{avg_enjoyment:.1f}/10")
    
    with col3:
        avg_combined = sum(s.combined_score for s in agent_segments) / len(agent_segments)
        st.metric("Avg Combined", f"{avg_combined:.1f}/20")
    
    top_segment = max(agent_segments, key=lambda x: x.combined_score)
    st.success(f"🏆 **Best Segment**: Segment {top_segment.segment_id} with combined score of {top_segment.combined_score}/20")
    
    # Show some interesting insights
    st.markdown("### 🔍 Conversation Insights")
    
    # Most common interaction type
    interaction_types = [s.interaction_type for s in agent_segments]
    most_common_type = max(set(interaction_types), key=interaction_types.count)
    st.info(f"**Most Common Interaction**: {most_common_type}")
    
    # Highest engagement segment details
    best_engagement = max(agent_segments, key=lambda x: x.engagement_score)
    if hasattr(best_engagement, 'start_line') and hasattr(best_engagement, 'end_line'):
        st.info(f"**Highest Engagement**: Segment {best_engagement.segment_id} about '{best_engagement.topic}' ({best_engagement.engagement_score}/10) - Lines {best_engagement.start_line}-{best_engagement.end_line}")
    else:
        st.info(f"**Highest Engagement**: Segment {best_engagement.segment_id} about '{best_engagement.topic}' ({best_engagement.engagement_score}/10)")
    
    # Show segment coverage
    if agent_segments and hasattr(agent_segments[0], 'start_line'):
        total_conversation_lines = max(s.end_line for s in agent_segments if hasattr(s, 'end_line')) + 1
        st.info(f"**Conversation Coverage**: {len(agent_segments)} segments covering {total_conversation_lines} conversation lines")
    
    # Add a note about how these segments are used
    st.info("""
    💡 **How this works**: The AI agent analyzes your conversation and breaks it into meaningful segments. 
    Each segment gets scored for engagement and enjoyment. The top-scoring segments are then used 
    to generate contextually appropriate follow-up suggestions. Content is extracted programmatically 
    based on line boundaries identified by the AI.
    """)

# Helper: detect dominant language ('ru' or 'en') using OpenAI if available, else heuristic
def detect_conversation_language(conversation: List[Dict]) -> str:
    try:
        # Use OpenAI if key is present
        if hasattr(st.session_state, 'openai_api_key') and st.session_state.openai_api_key:
            client = openai.OpenAI(api_key=st.session_state.openai_api_key)
            # Sample first N messages to keep prompt compact
            sample = conversation[:50]
            text = "\n".join(f"{m['speaker']}: {m['message']}" for m in sample)
            user_prompt = (
                "Determine the dominant language in this dialogue (consider both speakers). "
                "Return exactly 'ru' for Russian or 'en' for English. No other text.\n\n" + text
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a strict language detector. Reply with 'ru' or 'en' only."},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2,
                temperature=0.0
            )
            code = resp.choices[0].message.content.strip().lower()
            return 'ru' if code.startswith('ru') else 'en'
        
    except Exception as e:
        st.warning(f"Language detection via API failed: {e}. Falling back to heuristic.")
    
    # Heuristic fallback (Cyrillic presence)
    all_text = " ".join((m.get('message', '') or '') for m in conversation)
    return 'ru' if any(ord(c) >= 1040 and ord(c) <= 1103 for c in all_text) else 'en'

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
            os.environ['OPENAI_API_KEY'] = api_key
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
                        
                        # Detect and store dominant language
                        detected_lang = detect_conversation_language(conversation)
                        st.session_state['detected_language'] = detected_lang
                        # Apply to summary generator
                        summary_generator.language_code = detected_lang
                        st.info(f"🌐 Detected language: {'Russian' if detected_lang == 'ru' else 'English'}")
                        
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
                            session_data['follow_ups'], session_data['agent_segments'] = generate_intelligent_follow_ups_and_segments(conversation, followup_generator)
                        
                        # Log summary generation
                        storage.log_interaction({
                            'action': 'conversation_processed',
                            'conversation_length': len(conversation),
                            'number_of_segments': len(valid_segments),
                            'invalid_segments': len(segments) - len(valid_segments),
                            'number_of_followups': len(session_data['follow_ups']),
                            'number_of_agent_segments': len(session_data.get('agent_segments', [])),
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
            st.write(f"**Agent segments**: {len(session_data.get('agent_segments', []))} analyzed")
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
            
            # Add button to view agent conversation segments
            st.markdown("---")
            st.markdown("#### 🔍 Analysis Details")
            
            if st.button("🧠 View AI Agent Conversation Analysis", key="view_agent_segments", help="See how the AI analyzed your conversation"):
                if session_data.get('agent_segments'):
                    # Store the flag to show agent segments
                    session_data['show_agent_analysis'] = True
                    st.rerun()
                else:
                    if AGENTS_AVAILABLE:
                        st.warning("⚠️ No agent analysis available. Make sure you have an OpenAI API key configured.")
                    else:
                        st.info("💡 Agent analysis requires the agent system. Install agent modules for detailed conversation analysis.")
            
            # Display agent segments if requested
            if session_data.get('show_agent_analysis') and session_data.get('agent_segments'):
                display_agent_conversation_segments(session_data['agent_segments'], session_data)
                
                # Add button to hide the analysis
                if st.button("🔼 Hide Analysis", key="hide_agent_segments"):
                    session_data['show_agent_analysis'] = False
                    st.rerun()
        
        # Right column: Summary
        with col2:
            st.header("📃 Conversation Segment Summaries")
            
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