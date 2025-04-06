from typing import Dict

from pydantic import BaseModel

from agents.base_agent import BaseAgent
from prompts.router_prompt import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT


class RouterResponse(BaseModel):
    """
    The structure of the response from the router agent.
    """

    datasource: str


class Router(BaseAgent):
    """
    The Router class is responsible for routing the question
    to the appropriate agent. It uses a language model to
    determine which agent should handle the question.
    """

    def get_system_message(self) -> Dict[str, str]:
        return {
            "role": ROUTER_SYSTEM_PROMPT.role,
            "content": ROUTER_SYSTEM_PROMPT.format(),
        }

    def get_user_message(self, question: str) -> Dict[str, str]:
        return {
            "role": ROUTER_USER_PROMPT.role,
            "content": ROUTER_USER_PROMPT.format({"question": question}),
        }

    def generate_response(self, question: str) -> RouterResponse:
        system_message = self.get_system_message()
        user_message = self.get_user_message(question=question)

        response = self.chat_client.generate_structured_response(
            system_message=system_message,
            user_message=user_message,
            response_model=RouterResponse,
        )

        return response
