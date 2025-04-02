from pathlib import Path
from typing import Dict

from langchain_core.documents.base import Document
from pydantic import BaseModel

from ..api_clients.groq_chat_client import GroqChatClient
from ..prompts.retrieval_grader_prompt import (
    RETRIEVAL_SYSTEM_PROMPT,
    RETRIEVAL_USER_PROMPT,
)


class GraderResponse(BaseModel):
    score: str
    explanation: str


class RetrievalGrader:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.chat_client = GroqChatClient(config_path=config_path)

    def get_system_message(self) -> Dict[str, str]:
        system_message = {
            "role": RETRIEVAL_SYSTEM_PROMPT.role,
            "content": RETRIEVAL_SYSTEM_PROMPT.format(),
        }
        return system_message

    def get_user_message(self, question: str, document: str) -> Dict[str, str]:
        prompt_content = RETRIEVAL_USER_PROMPT.format(
            {"question": question, "document": document}
        )
        user_message = {
            "role": RETRIEVAL_USER_PROMPT.role,
            "content": prompt_content,
        }
        return user_message

    def generate_response(
        self, question: str, document: Document
    ) -> GraderResponse:
        system_message = self.get_system_message()
        user_message = self.get_user_message(question, document)

        return self.chat_client.generate_structured_response(
            system_message=system_message,
            user_message=user_message,
            response_model=GraderResponse,
        )
