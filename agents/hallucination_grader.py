from pathlib import Path
from typing import Dict

from pydantic import BaseModel

from ..api_clients.groq_chat_client import GroqChatClient
from ..prompts.hallucination_grader_prompt import (
    HALLUCINATION_SYSTEM_PROMPT,
    HALLUCINATION_USER_PROMPT,
)


class HallucinationGraderResponse(BaseModel):
    score: str


class HallucinationGrader:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.chat_client = GroqChatClient(config_path=config_path)

    def get_system_message(self) -> Dict[str, str]:
        system_message = {
            "role": HALLUCINATION_SYSTEM_PROMPT.role,
            "content": HALLUCINATION_SYSTEM_PROMPT.format(),
        }
        return system_message

    def get_user_message(
        self, documents: str, generation: str
    ) -> Dict[str, str]:
        prompt_content = HALLUCINATION_USER_PROMPT.format(
            {"documents": documents, "generation": generation}
        )
        user_message = {
            "role": HALLUCINATION_USER_PROMPT.role,
            "content": prompt_content,
        }
        return user_message

    def generate_response(
        self, documents: str, generation: str
    ) -> HallucinationGraderResponse:
        system_message = self.get_system_message()
        user_message = self.get_user_message(documents, generation)

        return self.chat_client.generate_structured_response(
            system_message=system_message,
            user_message=user_message,
            response_model=HallucinationGraderResponse,
        )
