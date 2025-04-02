from typing import Dict

from pydantic import BaseModel

from ..prompts.answer_grader_prompt import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT,
)
from .base_agent import BaseAgent


class AnswerGraderResponse(BaseModel):
    score: str


class AnswerGrader(BaseAgent):

    def get_system_message(self) -> Dict[str, str]:
        return {
            "role": ANSWER_SYSTEM_PROMPT.role,
            "content": ANSWER_SYSTEM_PROMPT.format(),
        }

    def get_user_message(
        self, generation: str, question: str
    ) -> Dict[str, str]:
        return {
            "role": ANSWER_USER_PROMPT.role,
            "content": ANSWER_USER_PROMPT.format(
                {"generation": generation, "question": question}
            ),
        }

    def generate_response(
        self, generation: str, question: str
    ) -> AnswerGraderResponse:
        system_message = self.get_system_message()
        user_message = self.get_user_message(
            generation=generation, question=question
        )

        response = self.chat_client.generate_structured_response(
            system_message=system_message,
            user_message=user_message,
            response_model=AnswerGraderResponse,
        )

        return response
