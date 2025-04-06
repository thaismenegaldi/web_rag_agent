from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, TypeVar, Union

from pydantic import BaseModel

from api_clients.groq_chat_client import GroqChatClient

T = TypeVar("T", bound=BaseModel)


class BaseAgent(ABC):
    """
    Base class for all agents.
    This class provides a common interface for all agents, including methods
    for generating system and user messages, as well as generating responses.
    Subclasses should implement the abstract methods to provide specific
    functionality.

    Parameters
    ----------
    config_path : Path
        Path to the configuration file for the agent. This file is used to
        initialize the GroqChatClient.
    """

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.chat_client = GroqChatClient(config_path=config_path)

    @abstractmethod
    def get_system_message(self) -> Dict[str, str]:
        """
        Returns the system message for the agent. This message is used to
        initialize the agent's behavior and capabilities.

        Returns
        -------
        Dict[str, str]
            The system message for the agent.
        """
        pass

    @abstractmethod
    def get_user_message(self, **kwargs) -> Dict[str, str]:
        """
        Returns the user message for the agent. This message is used to
        initialize the agent's input and context.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to customize the user message.

        Returns
        -------
        Dict[str, str]
            The user message for the agent.
        """
        pass

    @abstractmethod
    def generate_response(self, **kwargs) -> Union[str, T]:
        """
        Generates a response from the agent based on the provided input.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to customize the response generation.

        Returns
        -------
        str
            The generated response from the agent.
        """
        pass
