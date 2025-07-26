import streamlit as st
import re
import json
import os
import openai
import nltk
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import hashlib

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
    """Generate AI-powered summaries with sentence-to-source mapping."""
    
    def __init__(self):
        self.summary_prompt = """Could you please provide a summary of a given dialogue, including all key points and supporting details? The summary should be comprehensive and accurately reflect the main message and arguments presented in the original dialogue, while also being concise and easy to understand. To ensure accuracy, please read the text carefully and pay attention to any nuances or complexities in the language. Additionally, the summary should avoid any personal biases or interpretations and remain objective and factual throughout."""
        
    def format_conversation_for_prompt(self, conversation: List[Dict]) -> str:
        """Format conversation for AI prompt."""
        formatted = "DIALOGUE:\n"
        for msg in conversation:
            formatted += f"[{msg['timestamp']}] {msg['speaker']}: {msg['message']}\n"
        return formatted
    
    def generate_summary_with_api(self, conversation: List[Dict]) -> str:
        """Generate summary using OpenAI API."""
        try:
            # Check if API key is available
            if not hasattr(st.session_state, 'openai_api_key') or not st.session_state.openai_api_key:
                return self.generate_fallback_summary(conversation)
            
            client = openai.OpenAI(api_key=st.session_state.openai_api_key)
            
            conversation_text = self.format_conversation_for_prompt(conversation)
            full_prompt = f"{self.summary_prompt}\n\n{conversation_text}"
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate, objective summaries of dialogues."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return self.generate_fallback_summary(conversation)
    
    
    def map_sentences_to_sources(self, summary: str, conversation: List[Dict]) -> Dict[str, List[int]]:
        """Map each sentence in summary to relevant conversation messages."""
        sentences = nltk.sent_tokenize(summary)
        sentence_mapping = {}
        
        for i, sentence in enumerate(sentences):
            sentence_id = f"sent_{i}"
            relevant_messages = []
            
            # Simple keyword matching to find relevant messages
            sentence_words = set(sentence.lower().split())
            
            for j, msg in enumerate(conversation):
                msg_words = set(msg['message'].lower().split())
                
                # Calculate word overlap
                overlap = len(sentence_words.intersection(msg_words))
                overlap_ratio = overlap / max(len(sentence_words), 1)
                
                # If there's significant overlap, consider it relevant
                if overlap_ratio > 0.1 or overlap >= 2:
                    relevant_messages.append(j)
            
            # If no matches found, assign to nearby messages based on position
            if not relevant_messages:
                # Assign to middle portion of conversation as fallback
                mid_point = len(conversation) // 2
                relevant_messages = [max(0, mid_point - 1), mid_point, min(len(conversation) - 1, mid_point + 1)]
            
            sentence_mapping[sentence_id] = relevant_messages
        
        return sentence_mapping, sentences

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
                'summary': "",
                'summary_sentences': [],
                'sentence_mapping': {},
                'follow_ups': [],
                'selected_follow_up': None,
                'user_feedback': {},
                'selected_sentence': None
            }
        return st.session_state.session_data

def generate_summary_and_mapping(conversation: List[Dict], summary_generator: SummaryGenerator) -> Tuple[str, Dict, List[str]]:
    """Generate summary and create sentence-to-source mapping."""
    summary = summary_generator.generate_summary_with_api(conversation)
    sentence_mapping, sentences = summary_generator.map_sentences_to_sources(summary, conversation)
    return summary, sentence_mapping, sentences

def display_interactive_summary(summary_sentences: List[str], sentence_mapping: Dict, conversation: List[Dict], session_data: Dict):
    """Display an interactive summary with clickable sentences."""
    st.markdown("### 📝 Interactive Summary")
    st.markdown("*Click on any sentence to view the source dialogue segments*")
    
    # Display each sentence as a clickable button
    for i, sentence in enumerate(summary_sentences):
        sentence_id = f"sent_{i}"
        
        # Create a button for each sentence
        if st.button(
            sentence.strip(),
            key=f"summary_sent_{i}",
            help="Click to view source dialogue",
            use_container_width=True
        ):
            session_data['selected_sentence'] = sentence_id
            
            # Log the sentence interaction
            storage = DataStorage()
            storage.log_interaction({
                'action': 'summary_sentence_clicked',
                'sentence_id': sentence_id,
                'sentence_number': i + 1,
                'sentence_text': sentence.strip()[:100],  # Log first 100 chars
                'mapped_message_indices': sentence_mapping.get(sentence_id, [])
            })
        
        # Add some spacing
        st.write("")
    
    # Display source context if a sentence is selected
    if session_data.get('selected_sentence'):
        selected_id = session_data['selected_sentence']
        if selected_id in sentence_mapping:
            st.markdown("---")
            st.markdown("### 🔍 Source Dialogue Context")
            
            relevant_indices = sentence_mapping[selected_id]
            sentence_num = int(selected_id.split('_')[1]) + 1
            
            st.info(f"📌 Showing source context for Summary Sentence {sentence_num}")
            
            # Display relevant conversation messages
            for idx in relevant_indices:
                if 0 <= idx < len(conversation):
                    msg = conversation[idx]
                    # Highlight the relevant message with proper contrast
                    st.markdown(f"""
                    <div style="background-color: #e8f4fd; color: #1e1e1e; padding: 15px; margin: 8px 0; border-radius: 8px; border-left: 5px solid #0066cc; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <strong style="color: #0066cc;">[{msg['timestamp']}] {msg['speaker']}:</strong><br>
                        <span style="color: #2c3e50; line-height: 1.5;">{msg['message']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Add button to clear selection
            if st.button("🔄 Clear Selection", key="clear_selection"):
                session_data['selected_sentence'] = None
                st.rerun()

def generate_mock_follow_ups() -> List[str]:
    """Generate mock follow-ups for testing purposes."""
    return [
        "I hope you feel better soon! Is there anything specific you'd like me to help you with while you recover?",
        "Would you like me to look up some remedies for cold symptoms that might help you feel better?",
        "Since the weather is quite variable, would you like me to send you daily weather updates while you're feeling unwell?",
        "Have you been able to rest well? Good sleep is really important when fighting off a cold.",
        "Would you like me to remind you to take your medication or suggest some warm drinks that might soothe your throat?"
    ]

def main():
    """Main Streamlit application."""
    
    # Initialize data storage and summary generator
    storage = DataStorage()
    session_data = storage.get_session_data()
    summary_generator = SummaryGenerator()
    
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
                    with st.spinner("Generating AI-powered summary..."):
                        # Generate summary with sentence mapping
                        summary, sentence_mapping, sentences = generate_summary_and_mapping(
                            conversation, summary_generator
                        )
                        
                        session_data['summary'] = summary
                        session_data['summary_sentences'] = sentences
                        session_data['sentence_mapping'] = sentence_mapping
                        session_data['follow_ups'] = generate_mock_follow_ups()
                        
                        # Log summary generation
                        storage.log_interaction({
                            'action': 'summary_generated',
                            'conversation_length': len(conversation),
                            'summary_sentences': len(sentences),
                            'has_api_key': bool(hasattr(st.session_state, 'openai_api_key') and st.session_state.openai_api_key)
                        })
                        
                        st.success("✅ Conversation processed with AI summary!")
                        st.rerun()
            else:
                st.error("❌ No valid conversation messages found in the file")
        
        # Session info
        if session_data['conversation']:
            st.divider()
            st.subheader("📊 Session Info")
            st.write(f"Messages: {len(session_data['conversation'])}")
            st.write(f"Summary generated: {'✅' if session_data['summary'] else '❌'}")
            st.write(f"Summary sentences: {len(session_data.get('summary_sentences', []))}")
            st.write(f"Follow-ups generated: {'✅' if session_data['follow_ups'] else '❌'}")
    
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
            st.header("📝 Conversation Summary")
            
            if session_data['summary'] and session_data.get('summary_sentences'):
                # Display interactive summary
                display_interactive_summary(
                    session_data['summary_sentences'],
                    session_data['sentence_mapping'],
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
            
            elif session_data['summary']:
                # Fallback to regular summary display if sentences aren't available
                st.markdown("### 📝 Summary")
                st.markdown(session_data['summary'])
                st.info("💡 Process the conversation again to enable interactive summary features")
            
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