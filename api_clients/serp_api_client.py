import os
from typing import Dict

import requests
from dotenv import load_dotenv

# take environment variables
load_dotenv()


class SerpAPIClient:
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_KEY environment variable is missing.")

    def search_tool(self, query: str) -> Dict:

        # general google search
        url = f"https://serpapi.com/search.json?q={query}&hl=en&api_key={self.api_key}"  # noqa: E501

        # get response
        response = requests.get(url).json()

        return response
