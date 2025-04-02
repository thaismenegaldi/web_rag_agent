from ..prompts.question_answering_prompt import (
    QEA_SYSTEM_PROMPT, QEA_USER_PROMPT
)
from ..api_clients.groq_chat_client import GroqChatClient
from pathlib import Path
from typing import Dict


class RetrievalAugmentedGenerator:
    def __init__(self, retriever, config_path: Path) -> None:
        self.retriever = retriever
        self.config_path = config_path
        self.chat_client = GroqChatClient(config_path=config_path)

    def retrieve_context(self, question: str) -> str:
        context = self.retriever.invoke(question)
        return context if context else "No relevant context found."

    def get_system_message(self) -> Dict[str, str]:
        system_message = {
            "role": QEA_SYSTEM_PROMPT.role,
            "content": QEA_SYSTEM_PROMPT.format()
        }
        return system_message

    def get_user_message(self, question: str, context: str) -> Dict[str, str]:
        prompt_content = QEA_USER_PROMPT.format({
            "question": question,
            "context": context
        })
        user_message = {
            "role": QEA_USER_PROMPT.role,
            "content": prompt_content
        }
        return user_message

    def generate_response(self, question: str) -> str:
        context = self.retrieve_context(question)
        system_message = self.get_system_message()
        user_message = self.get_user_message(question, context)

        return self.chat_client.generate_response(system_message, user_message)
