# Agentic Follow-up Rating System - Implementation Plan

## Project Overview
Build a Streamlit web interface for evaluating and rating AI-generated conversation follow-ups to improve model training and performance.

## Objectives
- Create an intuitive interface for reviewing AI-generated follow-ups
- Enable data collection for model training and improvement
- Provide contextual summary analysis with source traceability
- Allow collaborative editing and feedback on follow-ups

## Technical Requirements

### Input Format
- Support text files containing timestamped conversation logs
- Handle format: `[timestamp] Speaker: message content`
- Parse conversations with multiple participants (Agent, User, etc.)

### Core Features
1. **Conversation Summary Generation**
   - Generate comprehensive summaries using specified prompt
   - Link each summary sentence to source dialogue segments
   - Enable click-to-view functionality for context verification

2. **Follow-up Generation & Rating**
   - Display 5 AI-generated follow-ups for evaluation
   - Provide editing capabilities for follow-up refinement
   - Collect user ratings and comments

3. **Data Collection**
   - Store user interactions into a log file
   - Track follow-up modifications and selection of a final follow-up
   - Export feedback data for analysis

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Set up Streamlit application framework
- [ ] Implement file upload and parsing functionality
- [ ] Create basic UI layout (left: follow-ups, right: summary)
- [ ] Establish data storage structure

### Phase 2: Summary System
- [ ] Integrate summary generation prompt:
  ```
  "Could you please provide a summary of a given dialogue, including all key points and supporting details? The summary should be comprehensive and accurately reflect the main message and arguments presented in the original dialgue, while also being concise and easy to understand. To ensure accuracy, please read the text carefully and pay attention to any nuances or complexities in the language. Additionally, the summary should avoid any personal biases or interpretations and remain objective and factual throughout."
  ```
- [ ] Implement sentence-to-source mapping
- [ ] Create interactive summary display with clickable segments
- [ ] Add context popup/sidebar for source dialogue

### Phase 3: Follow-up Management
- [ ] Develop follow-up generation system
- [ ] Create editable follow-up interface
- [ ] Implement rating system (1-5 stars, thumbs up/down, etc.)
- [ ] Add comment functionality for detailed feedback

### Phase 4: Data Collection & Export
- [ ] Implement feedback data storage
- [ ] Create export functionality for training data
- [ ] Add analytics dashboard for evaluation metrics
- [ ] Establish data validation and quality checks

### Phase 5: Enhancement & Optimization
- [ ] Optimize UI/UX based on user feedback
- [ ] Add batch processing capabilities
- [ ] Implement user authentication if needed
- [ ] Add conversation history and session management

## Technical Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **Data Storage**: JSON/CSV files or lightweight database
- **AI Integration**: API calls for summary/follow-up generation

## Success Metrics
- User engagement time per conversation
- Quality of collected feedback data
- Improvement in follow-up relevance scores
- Efficiency of annotation process

## Sample Input Reference
```
[4/18/2025, 12:29:52 AM] Agent: What's new, [REDACTED]?
[4/18/2025, 12:29:57 AM] User 2: I have a very bad cold,
[4/18/2025, 12:30:01 AM] Agent: Oh, get well soon! Let me check the weather so you know how to dress warmly.
[4/18/2025, 12:30:04 AM] Agent: Thunderstorms near Dundas, Minnesota.
[4/18/2025, 12:30:23 AM] Agent: It's 24 degrees Celsius or 75 degrees Fahrenheit in Miami right now, clear, but with a 17 kilometer per hour northeast wind. It's warm, but it's best to be careful if you have a cold.  Maybe some tea with lemon and honey? I know in Georgian honey is called "tapli" - you probably have the most delicious mountain honey, right?
[4/18/2025, 12:30:30 AM] User 2: Yes, I agree.
...
```