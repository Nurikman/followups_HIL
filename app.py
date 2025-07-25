import streamlit as st
import re
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional

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
                'follow_ups': [],
                'selected_follow_up': None,
                'user_feedback': {}
            }
        return st.session_state.session_data

def generate_mock_summary(conversation: List[Dict]) -> str:
    """Generate a mock summary for testing purposes."""
    if not conversation:
        return "No conversation to summarize."
    
    participants = set(item['speaker'] for item in conversation)
    message_count = len(conversation)
    
    return f"""Summary of conversation between {', '.join(participants)}:

The conversation consists of {message_count} messages. The discussion appears to involve health-related concerns and weather information. Key topics include illness, weather conditions, and helpful suggestions for care.

This is a placeholder summary that will be replaced with AI-generated content in Phase 2."""

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
    
    # Initialize data storage
    storage = DataStorage()
    session_data = storage.get_session_data()
    
    # Header
    st.title("🤖 Agentic Follow-up Rating System")
    st.markdown("Rate and improve AI-generated conversation follow-ups")
    
    # Sidebar for file upload and controls
    with st.sidebar:
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
                    with st.spinner("Processing conversation..."):
                        session_data['summary'] = generate_mock_summary(conversation)
                        session_data['follow_ups'] = generate_mock_follow_ups()
                        st.success("✅ Conversation processed!")
                        st.rerun()
            else:
                st.error("❌ No valid conversation messages found in the file")
        
        # Session info
        if session_data['conversation']:
            st.divider()
            st.subheader("📊 Session Info")
            st.write(f"Messages: {len(session_data['conversation'])}")
            st.write(f"Summary generated: {'✅' if session_data['summary'] else '❌'}")
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
            
            if session_data['summary']:
                st.markdown(session_data['summary'])
                
                st.subheader("🔍 Summary Analysis")
                st.info("💡 **Coming in Phase 2**: Click on summary sentences to view source dialogue segments")
                
                # Show selected follow-up
                if session_data.get('selected_follow_up') is not None:
                    st.subheader("✅ Selected Follow-up")
                    selected_idx = session_data['selected_follow_up']
                    selected_text = session_data['user_feedback'][f'follow_up_{selected_idx}']['edited_text']
                    st.success(f"**Follow-up {selected_idx + 1}**: {selected_text}")
            
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