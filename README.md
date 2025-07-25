# Agentic Follow-up Rating System

A Streamlit web interface for evaluating and rating AI-generated conversation follow-ups to improve model training and performance.

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd followups_HIL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Usage

1. **Upload Conversation**: Use the sidebar to upload a text file containing conversation logs
2. **Process**: Click "Process Conversation" to generate summary and follow-ups
3. **Review**: Evaluate the 5 generated follow-ups on the left panel
4. **Edit**: Modify follow-ups as needed using the text areas
5. **Rate & Select**: Rate each follow-up and select the best one
6. **Feedback**: Add comments to provide detailed feedback

## 📋 Input Format

Upload text files with conversations in this format:
```
[4/18/2025, 12:29:52 AM] Agent: What's new, [REDACTED]?
[4/18/2025, 12:29:57 AM] User 2: I have a very bad cold,
[4/18/2025, 12:30:01 AM] Agent: Oh, get well soon! Let me check the weather so you know how to dress warmly.
```

## 🗄️ Data Collection

User interactions are automatically logged to `user_interactions.jsonl` for model training purposes.

## 🔄 Implementation Status

- ✅ **Phase 1**: Core Infrastructure (Current)
  - Streamlit application framework
  - File upload and parsing functionality
  - Basic UI layout (left: follow-ups, right: summary)
  - Data storage structure

- 🔲 **Phase 2**: Summary System (Planned)
  - AI-powered summary generation
  - Sentence-to-source mapping
  - Interactive summary display

- 🔲 **Phase 3**: Follow-up Management (Planned)
  - AI follow-up generation
  - Advanced rating system
  - Enhanced feedback collection

## 📝 Current Features

- **File Upload**: Support for text conversation files
- **Conversation Parsing**: Automatic parsing of timestamped messages
- **Mock Summary**: Placeholder summary generation (Phase 2 will add AI)
- **Follow-up Rating**: 5-point rating system for each follow-up
- **Editing**: Inline editing of follow-ups
- **Selection**: Choose the best follow-up from options
- **Feedback**: Comment system for detailed feedback
- **Data Logging**: Automatic logging of user interactions

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Data Storage**: JSONL files
- **Future**: AI integration for summary/follow-up generation

