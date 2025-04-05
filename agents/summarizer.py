from typing import Dict, KeysView, List, Union

from prompts.summarizer_prompt import (
    SUMMARIZER_SYSTEM_PROMPT,
    SUMMARIZER_USER_PROMPT,
)
from agents.base_agent import BaseAgent


class Summarizer(BaseAgent):

    def get_system_message(self) -> Dict[str, str]:
        return {
            "role": SUMMARIZER_SYSTEM_PROMPT.role,
            "content": SUMMARIZER_SYSTEM_PROMPT.format(),
        }

    def get_user_message(
        self, question: str, context: Union[KeysView[str], List[str]]
    ) -> Dict[str, str]:
        return {
            "role": SUMMARIZER_USER_PROMPT.role,
            "content": SUMMARIZER_USER_PROMPT.format(
                {"question": question, "context": context}
            ),
        }

    def generate_response(
        self, question: str, context: Union[Dict, str, List[Dict]]
    ) -> str:
        system_message = self.get_system_message()
        user_message = self.get_user_message(
            question=question,
            context=context,
        )

        response = self.chat_client.generate_response(
            system_message=system_message,
            user_message=user_message,
        )

        return response
