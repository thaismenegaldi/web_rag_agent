import os
from pathlib import Path
from typing import Type, TypeVar

import groq
import instructor
import yaml
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

# take environment variables
load_dotenv()

T = TypeVar("T", bound=BaseModel)


class GroqChatClient:
    def __init__(self, config_path: Path):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")

        self.groq_client = groq.Client(api_key=api_key)
        self.groq_instructor = instructor.from_groq(
            Groq(), mode=instructor.Mode.JSON
        )

        self.config = self._load_yaml_config(config_path)

    @staticmethod
    def _load_yaml_config(config_path: Path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            return config

    def generate_response(self, system_message: str, user_message: str) -> str:
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
