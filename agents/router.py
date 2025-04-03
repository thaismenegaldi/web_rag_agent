from typing import Dict

from pydantic import BaseModel

from ..prompts.router_prompt import (
    ROUTER_SYSTEM_PROMPT,
    ROUTER_USER_PROMPT,
)
from .base_agent import BaseAgent


class RouterResponse(BaseModel):
    datasource: str


class Router(BaseAgent):

    def get_system_message(self) -> Dict[str, str]:
        return {
            "role": ROUTER_SYSTEM_PROMPT.role,
            "content": ROUTER_SYSTEM_PROMPT.format(),
        }

    def get_user_message(
        self, question: str
    ) -> Dict[str, str]:
        return {
            "role": ROUTER_USER_PROMPT.role,
            "content": ROUTER_USER_PROMPT.format(
                {"question": question}
            ),
        }

    def generate_response(
        self, question: str
    ) -> RouterResponse:
        system_message = self.get_system_message()
        user_message = self.get_user_message(
            question=question
        )

        response = self.chat_client.generate_structured_response(
            system_message=system_message,
            user_message=user_message,
            response_model=RouterResponse,
        )

        return response
