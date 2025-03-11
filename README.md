# AI Conversation Agent with Memory

This project implements an AI conversation agent with both short-term and long-term memory capabilities. The agent can maintain context throughout conversations and recall previous interactions using semantic search.

## Features

-   **Short-term Memory**: Maintains recent conversation context with configurable capacity
-   **Long-term Memory**: Stores and retrieves memories using semantic search (FAISS)
-   **Contextual Understanding**: Processes messages with additional context information
-   **Personality Traits**: Configurable personality settings
-   **Conversation Summaries**: Ability to generate summaries of recent conversations
-   **OpenAI Integration**: Uses GPT-4 for generating human-like responses

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
    - Copy the `.env.example` file to `.env`
    - Replace `your_api_key_here` with your actual OpenAI API key
    ```bash
    cp .env.example .env
    # Edit .env and add your OpenAI API key
    ```

## Usage

### Basic Example

```python
from src.agent import ConversationAgent

# Initialize the agent
agent = ConversationAgent(
    personality={
        "name": "Sam",
        "tone": "friendly and helpful",
        "interests": "helping with tasks and learning from conversations",
        "communication_style": "clear and engaging"
    },
    openai_model="gpt-4-turbo-preview"  # Optional: specify which OpenAI model to use
)

# Process a message
response = agent.process_message(
    "Hi! My name is Alice.",
    context={"user": "Alice", "timestamp": "morning"}
)
print(response)

# Get conversation summary
print(agent.get_conversation_summary())
```

### Running the Example Script

```bash
python src/example.py
```

## Components

### Memory System

-   `ShortTermMemory`: Manages recent conversations with a fixed capacity
-   `LongTermMemory`: Stores historical conversations with semantic search capabilities
-   `Memory`: Data class for storing individual memories with metadata

### ConversationAgent

The main agent class that:

-   Processes incoming messages
-   Maintains conversation context
-   Generates responses using OpenAI's GPT models
-   Manages personality traits
-   Provides conversation summaries

## Customization

### Adjusting Memory Capacity

```python
agent = ConversationAgent(short_term_capacity=20)  # Default is 10
```

### Updating Personality

```python
agent.update_personality({
    "tone": "professional",
    "interests": "technical discussions"
})
```

### Changing OpenAI Model

```python
agent = ConversationAgent(openai_model="gpt-3.5-turbo")  # Use GPT-3.5 instead of GPT-4
```

## Environment Variables

The following environment variables can be configured in the `.env` file:

-   `OPENAI_API_KEY`: Your OpenAI API key (required)

## Note

Make sure to keep your OpenAI API key secure and never commit it to version control. The `.env` file is included in `.gitignore` by default.
