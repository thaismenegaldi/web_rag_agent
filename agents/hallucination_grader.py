from typing import Dict

from pydantic import BaseModel

from agents.base_agent import BaseAgent
from prompts.hallucination_grader_prompt import (
    HALLUCINATION_SYSTEM_PROMPT,
    HALLUCINATION_USER_PROMPT,
)


class HallucinationGraderResponse(BaseModel):
    """
    The structure of the response from the hallucination grader agent.
    """

    score: str


class HallucinationGrader(BaseAgent):
    """
    The HallucinationGrader class is responsible for grading the hallucination
    of a given generation based on the provided documents. It uses a chat client
    to interact with a language model and generate a structured response.
    """

    def get_system_message(self) -> Dict[str, str]:
        return {
            "role": HALLUCINATION_SYSTEM_PROMPT.role,
            "content": HALLUCINATION_SYSTEM_PROMPT.format(),
        }

    def get_user_message(
        self, documents: str, generation: str
    ) -> Dict[str, str]:
        return {
            "role": HALLUCINATION_USER_PROMPT.role,
            "content": HALLUCINATION_USER_PROMPT.format(
                {"documents": documents, "generation": generation}
            ),
        }

    def generate_response(
        self, documents: str, generation: str
    ) -> HallucinationGraderResponse:
        system_message = self.get_system_message()
        user_message = self.get_user_message(documents, generation)

        response = self.chat_client.generate_structured_response(
            system_message=system_message,
            user_message=user_message,
            response_model=HallucinationGraderResponse,
        )

        return response
