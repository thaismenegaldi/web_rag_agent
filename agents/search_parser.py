from typing import Dict, KeysView, List, Union

from pydantic import BaseModel

from prompts.search_parser_promt import (
    SEARCH_SYSTEM_PROMPT,
    SEARCH_USER_PROMPT,
)
from agents.base_agent import BaseAgent


class ParserResponse(BaseModel):
    field: str


class SearchParser(BaseAgent):

    def get_system_message(self) -> Dict[str, str]:
        return {
            "role": SEARCH_SYSTEM_PROMPT.role,
            "content": SEARCH_SYSTEM_PROMPT.format(),
        }

    def get_user_message(
        self, question: str, context: Union[KeysView[str], List[str]]
    ) -> Dict[str, str]:
        return {
            "role": SEARCH_USER_PROMPT.role,
            "content": SEARCH_USER_PROMPT.format(
                {"question": question, "context": context}
            ),
        }

    def generate_response(
        self, question: str, context: Union[KeysView[str], List[str]]
    ) -> ParserResponse:
        system_message = self.get_system_message()
        user_message = self.get_user_message(
            question=question,
            context=context,
        )

        response = self.chat_client.generate_structured_response(
            system_message=system_message,
            user_message=user_message,
            response_model=ParserResponse,
        )

        return response
