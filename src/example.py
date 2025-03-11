from agent import ConversationAgent


def main():
    # Initialize the agent
    agent = ConversationAgent(
        personality={
            "name": "Sam",
            "tone": "friendly and helpful",
            "interests": "helping with tasks and learning from conversations",
            "communication_style": "clear and engaging",
        }
    )

    # Example conversation
    messages = [
        "Hi! My name is Alice. Nice to meet you!",
        "What kind of topics do you enjoy discussing?",
        "Can you tell me what we talked about earlier?",
        "What's your favorite part about our conversation so far?",
    ]

    print("Starting conversation simulation...")
    print("-" * 50)

    for message in messages:
        print(f"\nUser: {message}")

        # Process message with some context
        response = agent.process_message(
            message, context={"user": "Alice", "timestamp": "morning"}
        )

        print(f"Agent: {response}")

        # After a few messages, show the conversation summary
        if message == messages[-2]:
            print("\nGenerating conversation summary...")
            print(agent.get_conversation_summary())


if __name__ == "__main__":
    main()
