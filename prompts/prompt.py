from typing import Dict, Literal
from pydantic import BaseModel


class Prompt(BaseModel):
    role: Literal["system", "user"]
    name: str
    prompt_template: str

    def format(self, fields: Dict[str, str] = None) -> str:
        fields = fields or {}
        return self.prompt_template.format(**fields)
