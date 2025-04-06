import os
from pathlib import Path
from typing import Type, TypeVar

import groq
import instructor
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

from utils.load_config import load_yaml_config

# take environment variables
load_dotenv()

T = TypeVar("T", bound=BaseModel)


class GroqChatClient:
    """
    GroqChatClient is a client for interacting with the Groq API.
    It uses the Groq Python client to send chat completions requests
    and receive responses.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file containing API settings.
    """

    def __init__(self, config_path: Path) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")

        self.groq_client = groq.Client(api_key=api_key)
        self.groq_instructor = instructor.from_groq(
            Groq(), mode=instructor.Mode.JSON
        )
        self.config = load_yaml_config(config_path)

    def generate_response(self, system_message: str, user_message: str) -> str:
        """
        Generate a response from the Groq API using the provided system
        and user messages.

        Parameters
        ----------
        system_message : str
            The system message to be sent to the API.
        user_message : str
            The user message to be sent to the API.

        Returns
        -------
        str
            The generated response from the API.
        """
        try:
            response = self.groq_client.chat.completions.create(
                model=self.config["llm"]["model"],
                messages=[system_message, user_message],
                temperature=self.config["llm"]["temperature"],
                top_p=self.config["llm"]["top_p"],
                max_tokens=self.config["llm"]["max_tokens"],
            )

            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}")

    def generate_structured_response(
        self, system_message: str, user_message: str, response_model: Type[T]
    ) -> T:
        """
        Generate a structured response from the Groq API using the provided
        system and user messages.

        Parameters
        ----------
        system_message : str
            The system message to be sent to the API.
        user_message : str
            The user message to be sent to the API.
        response_model : Type[T]
            Describes the expected structure of the response. This should be a
            Pydantic model that defines the schema of the expected response.

        Returns
        -------
        response : Type[T]
            The generated response from the API structured as the response
            model.
        """
        try:
            response = self.groq_instructor.chat.completions.create(
                model=self.config["llm"]["model"],
                messages=[system_message, user_message],
                temperature=self.config["llm"]["temperature"],
                top_p=self.config["llm"]["top_p"],
                max_tokens=self.config["llm"]["max_tokens"],
                response_model=response_model,
            )

            return response
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}")
