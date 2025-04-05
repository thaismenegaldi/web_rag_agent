from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

from api_clients.groq_chat_client import GroqChatClient


class BaseAgent(ABC):
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.chat_client = GroqChatClient(config_path=config_path)

    @abstractmethod
    def get_system_message(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def get_user_message(self, **kwargs) -> Dict[str, str]:
        pass

    @abstractmethod
    def generate_response(self, **kwargs):
        pass
