from enum import Enum
from typing import Dict, Optional


class ConversationMove(Enum):
    FLATTER = "flatter"
    CHALLENGE = "challenge"
    DEFEND = "defend"
    DEFUSE = "defuse"
    ATTACK = "attack"
    ENTRUST = "entrust"


class ConversationMoves:
    MOVE_DESCRIPTIONS = {
        ConversationMove.FLATTER: {
            "description": "Express admiration or praise for the other agent's qualities, achievements, or ideas",
            "effect": "Builds rapport and trust, may make the other agent more receptive",
            "example": "Your experience in [field] is truly impressive, and your insights about [topic] are fascinating.",
        },
        ConversationMove.CHALLENGE: {
            "description": "Question or probe the other agent's assumptions, beliefs, or statements",
            "effect": "Stimulates deeper discussion and critical thinking",
            "example": "That's an interesting perspective, but have you considered [alternative viewpoint]?",
        },
        ConversationMove.DEFEND: {
            "description": "Protect and justify one's position, beliefs, or statements when challenged",
            "effect": "Maintains credibility and conviction in one's stance",
            "example": "I stand by my view because [reasoning], and my experience has shown that [evidence].",
        },
        ConversationMove.DEFUSE: {
            "description": "Reduce tension or conflict by finding common ground or redirecting the conversation",
            "effect": "Prevents escalation and maintains productive dialogue",
            "example": "I see where you're coming from, and perhaps we can find middle ground on [aspect].",
        },
        ConversationMove.ATTACK: {
            "description": "Directly criticize or oppose the other agent's position, beliefs, or statements",
            "effect": "Creates conflict and forces the other agent to defend or reconsider",
            "example": "Your argument is flawed because [reason], and here's why that matters...",
        },
        ConversationMove.ENTRUST: {
            "description": "Share personal or vulnerable information to build trust and deepen connection",
            "effect": "Creates intimacy and encourages reciprocal sharing",
            "example": "Let me share a personal experience about [topic] that shaped my perspective...",
        },
    }

    @staticmethod
    def get_move_description(move: ConversationMove) -> Dict:
        """Get the description and details of a conversation move."""
        return ConversationMoves.MOVE_DESCRIPTIONS[move]

    @staticmethod
    def format_move_for_prompt(
        move: ConversationMove, context: Optional[Dict] = None
    ) -> str:
        """Format a move description for inclusion in an AI prompt."""
        move_info = ConversationMoves.MOVE_DESCRIPTIONS[move]
        formatted = f"MOVE: {move.value.upper()}\n"
        formatted += f"Description: {move_info['description']}\n"
        formatted += f"Effect: {move_info['effect']}\n"
        formatted += f"Example: {move_info['example']}\n"

        if context:
            formatted += f"Context: {context.get('context', '')}\n"

        return formatted
