from datetime import datetime
import json
from pathlib import Path


class ConversationLogger:
    def __init__(self, session_id, agent1_name, agent2_name, topic):
        self.logs_dir = Path(__file__).parent.parent / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}_{session_id}.json"
        self.log_file = self.logs_dir / filename

        self.conversation_data = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "agent1": agent1_name,
            "agent2": agent2_name,
            "topic": topic,
            "messages": [],
        }
        self.save_log()

    def log_message(self, message_type, agent_name=None, content=None):
        message_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": message_type,
            "agent": agent_name,
            "content": content,
        }
        self.conversation_data["messages"].append(message_entry)
        self.save_log()

    def save_log(self):
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.conversation_data, f, indent=2, ensure_ascii=False)

    def log_summary(self, agent1_summary, agent2_summary):
        self.conversation_data["end_time"] = datetime.now().isoformat()
        self.conversation_data["summaries"] = {
            self.conversation_data["agent1"]: agent1_summary,
            self.conversation_data["agent2"]: agent2_summary,
        }
        self.save_log()
