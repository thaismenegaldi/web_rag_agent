from prompts.question_answering_prompt import (
    QEA_SYSTEM_PROMPT, QEA_USER_PROMPT
)
from agents.base_agent import BaseAgent
from pathlib import Path
from typing import Dict, List, Union
from langchain.schema import Document


class RetrievalAugmentedGenerator(BaseAgent):
    def __init__(self, retriever, config_path: Path) -> None:
        super().__init__(config_path)
        self.retriever = retriever

    def retrieve_context(self, question: str) -> str:
        context = self.retriever.invoke(question)
        return context if context else "No relevant context found."

    def get_system_message(self) -> Dict[str, str]:
        return {
            "role": QEA_SYSTEM_PROMPT.role,
            "content": QEA_SYSTEM_PROMPT.format()
        }

    def get_user_message(self, question: str, context: Union[Document, List]) -> Dict[str, str]:
        return {
            "role": QEA_USER_PROMPT.role,
            "content": QEA_USER_PROMPT.format({
                "question": question,
                "context": context
            })
        }

    def generate_response(self, question: str, context: Union[Document, List]) -> str:
        context = self.retrieve_context(question)
        system_message = self.get_system_message()
        user_message = self.get_user_message(question, context)

        return self.chat_client.generate_response(system_message, user_message)
