from pathlib import Path
from typing import Dict

from pydantic import BaseModel

from ..api_clients.groq_chat_client import GroqChatClient
from ..prompts.answer_grader_prompt import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT,
)


class AnswerGraderResponse(BaseModel):
    score: str


class AnswerGrader:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.chat_client = GroqChatClient(config_path=config_path)

    def get_system_message(self) -> Dict[str, str]:
        system_message = {
            "role": ANSWER_SYSTEM_PROMPT.role,
            "content": ANSWER_SYSTEM_PROMPT.format(),
        }
        return system_message

    def get_user_message(
        self, generation: str, question: str
    ) -> Dict[str, str]:
        prompt_content = ANSWER_USER_PROMPT.format(
            {"generation": generation, "question": question}
        )
        user_message = {
            "role": ANSWER_USER_PROMPT.role,
            "content": prompt_content,
        }
        return user_message

    def generate_response(
        self, generation: str, question: str
    ) -> AnswerGraderResponse:
        system_message = self.get_system_message()
        user_message = self.get_user_message(generation, question)

        return self.chat_client.generate_structured_response(
            system_message=system_message,
            user_message=user_message,
            response_model=AnswerGraderResponse,
        )
