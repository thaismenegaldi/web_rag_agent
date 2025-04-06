from typing import Dict

from langchain_core.documents.base import Document
from pydantic import BaseModel

from agents.base_agent import BaseAgent
from prompts.retrieval_grader_prompt import (
    RETRIEVAL_SYSTEM_PROMPT,
    RETRIEVAL_USER_PROMPT,
)


class GraderResponse(BaseModel):
    score: str
    explanation: str


class RetrievalGrader(BaseAgent):

    def get_system_message(self) -> Dict[str, str]:
        return {
            "role": RETRIEVAL_SYSTEM_PROMPT.role,
            "content": RETRIEVAL_SYSTEM_PROMPT.format(),
        }

    def get_user_message(
        self, question: str, document: Document
    ) -> Dict[str, str]:
        return {
            "role": RETRIEVAL_USER_PROMPT.role,
            "content": RETRIEVAL_USER_PROMPT.format(
                {"question": question, "document": document}
            ),
        }

    def generate_response(
        self, question: str, document: Document
    ) -> GraderResponse:
        system_message = self.get_system_message()
        user_message = self.get_user_message(question, document)

        response = self.chat_client.generate_structured_response(
            system_message=system_message,
            user_message=user_message,
            response_model=GraderResponse,
        )

        return response
