from typing import Dict, Optional
import yaml
from pathlib import Path


class ConversationMoves:
    _instance = None
    _move_descriptions = None
    _valid_moves = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConversationMoves, cls).__new__(cls)
            cls._load_moves()
        return cls._instance

    @classmethod
    def _load_moves(cls):
        if cls._move_descriptions is None:
            config_path = Path(__file__).parent / "config" / "moves.yaml"
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            cls._move_descriptions = config["moves"]
            cls._valid_moves = set(cls._move_descriptions.keys())

    @classmethod
    def get_valid_moves(cls) -> set[str]:
        """Get the set of valid move names."""
        cls._load_moves()
        return cls._valid_moves

    @staticmethod
    def get_move_description(move: str) -> Dict:
        """Get the description and details of a conversation move."""
        ConversationMoves._load_moves()
        if move not in ConversationMoves._valid_moves:
            raise ValueError(
                f"Invalid move: {move}. Valid moves are: {ConversationMoves._valid_moves}"
            )
        return ConversationMoves._move_descriptions[move]

    @staticmethod
    def format_move_for_prompt(move: str, context: Optional[Dict] = None) -> str:
        """Format a move description for inclusion in an AI prompt."""
        ConversationMoves._load_moves()
        if move not in ConversationMoves._valid_moves:
            raise ValueError(
                f"Invalid move: {move}. Valid moves are: {ConversationMoves._valid_moves}"
            )

        move_info = ConversationMoves._move_descriptions[move]
        formatted = f"MOVE: {move.upper()}\n"
        formatted += f"Description: {move_info['description']}\n"
        formatted += f"Effect: {move_info['effect']}\n"
        formatted += f"Example: {move_info['example']}\n"

        if context:
            formatted += f"Context: {context.get('context', '')}\n"

        return formatted
