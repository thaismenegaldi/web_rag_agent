from typing import Dict, Literal

from pydantic import BaseModel


class Prompt(BaseModel):
    """
    Represents a message prompt used in chat-based models.
    """

    role: Literal["system", "user"]
    name: str
    prompt_template: str

    def format(self, fields: Dict[str, str] = None) -> str:
        """
        Format the prompt template by filling in the placeholders with the
        given fields.

        Parameters
        ----------
        fields : dict of str to str, optional
            A dictionary of placeholder names and their corresponding values.
            If not provided, an empty dictionary is used.

        Returns
        -------
        str
            The formatted prompt string.
        """
        fields = fields or {}
        return self.prompt_template.format(**fields)
