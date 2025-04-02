import os
import yaml
from dotenv import load_dotenv
import groq
from pathlib import Path

# take environment variables
load_dotenv()


class GroqChatClient:
    def __init__(self, config_path: Path):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")

        self.groq_client = groq.Client(api_key=api_key)
        self.config = self._load_yaml_config(config_path)

    @staticmethod
    def _load_yaml_config(config_path: Path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            return config

    def generate_response(self, system_message: str, user_message: str):
        try:
            response = self.groq_client.chat.completions.create(
                model=self.config["llm"]["model"],
                messages=[
                    system_message,
                    user_message
                ],
                temperature=self.config["llm"]["temperature"],
                top_p=self.config["llm"]["top_p"],
                max_tokens=self.config["llm"]["max_tokens"]
            )

            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}")
