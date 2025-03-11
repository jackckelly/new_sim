from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
from agent import ConversationAgent
from datetime import datetime
import random
import yaml
import os

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)


def load_yaml_config(file_path):
    """Load configuration from a YAML file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, "r") as file:
        return yaml.safe_load(file)


# Load configurations
try:
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    agents_config = load_yaml_config(os.path.join(config_dir, "agents.yaml"))
    topics_config = load_yaml_config(os.path.join(config_dir, "topics.yaml"))

    AGENTS = agents_config["agents"]
    TOPICS = topics_config["topics"]
except Exception as e:
    print(f"Error loading configurations: {str(e)}")
    # Fallback to default configurations
    AGENTS = {
        "alice": {
            "name": "Alice",
            "tone": "friendly and curious",
            "interests": "tech startups, sci-fi movies, and future trends",
            "communication_style": "casual and enthusiastic",
        }
    }
    TOPICS = [
        "What's your take on AI and creativity?",
        "How do you like to spend your free time?",
    ]

# Store active simulations
active_simulations = {}


@app.route("/")
def index():
    # Get the first two agents as default selected agents
    selected_agents = list(AGENTS.keys())[:2]

    return render_template(
        "index.html",
        agents=AGENTS,
        topics=TOPICS,
        agent1=selected_agents[0],
        agent2=selected_agents[1],
    )


@socketio.on("start_simulation")
def handle_start_simulation(data):
    """Start a new conversation simulation between two agents"""
    session_id = request.sid
    agent1_id = data.get("agent1")
    agent2_id = data.get("agent2")
    topic = data.get("topic")

    if not all([agent1_id, agent2_id, topic]):
        emit("error", {"message": "Missing required parameters"})
        return

    # Initialize agents
    agent1 = ConversationAgent(personality=AGENTS[agent1_id], openai_model="gpt-4o")
    agent2 = ConversationAgent(personality=AGENTS[agent2_id], openai_model="gpt-4o")

    # Store simulation data
    active_simulations[session_id] = {
        "agent1": agent1,
        "agent2": agent2,
        "current_message": topic,
        "turn": 0,
        "is_active": True,
    }

    # Send initial topic
    emit("simulation_message", {"type": "topic", "content": topic})

    # Start the conversation
    continue_simulation(session_id)


@socketio.on("stop_simulation")
def handle_stop_simulation():
    """Stop the active simulation"""
    session_id = request.sid
    if session_id in active_simulations:
        active_simulations[session_id]["is_active"] = False

        # Get summaries
        agent1 = active_simulations[session_id]["agent1"]
        agent2 = active_simulations[session_id]["agent2"]

        emit(
            "simulation_message",
            {
                "type": "summary",
                "content": {
                    "agent1": agent1.get_conversation_summary(),
                    "agent2": agent2.get_conversation_summary(),
                },
            },
        )

        # Cleanup
        del active_simulations[session_id]


def continue_simulation(session_id):
    """Continue the conversation simulation"""
    if (
        session_id not in active_simulations
        or not active_simulations[session_id]["is_active"]
    ):
        return

    sim_data = active_simulations[session_id]
    current_message = sim_data["current_message"]
    turn = sim_data["turn"]

    # Agent 1's turn
    emit(
        "simulation_message",
        {"type": "thinking", "agent": sim_data["agent1"].personality["name"]},
    )

    response1 = sim_data["agent1"].process_message(
        current_message,
        context={
            "speaker": sim_data["agent2"].personality["name"],
            "turn": turn,
            "timestamp": datetime.now().isoformat(),
        },
    )

    emit(
        "simulation_message",
        {
            "type": "response",
            "agent": sim_data["agent1"].personality["name"],
            "content": response1,
        },
    )

    # Small delay between responses
    socketio.sleep(2)

    if not sim_data["is_active"]:
        return

    # Agent 2's turn
    emit(
        "simulation_message",
        {"type": "thinking", "agent": sim_data["agent2"].personality["name"]},
    )

    response2 = sim_data["agent2"].process_message(
        response1,
        context={
            "speaker": sim_data["agent1"].personality["name"],
            "turn": turn,
            "timestamp": datetime.now().isoformat(),
        },
    )

    emit(
        "simulation_message",
        {
            "type": "response",
            "agent": sim_data["agent2"].personality["name"],
            "content": response2,
        },
    )

    # Update simulation state
    sim_data["current_message"] = response2
    sim_data["turn"] += 1

    # Continue the simulation after a delay
    socketio.sleep(2)
    if sim_data["is_active"]:
        continue_simulation(session_id)


if __name__ == "__main__":
    socketio.run(app, debug=True)
