import os
import requests
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

class FirecrawlWebSearchToolInput(BaseModel):
    """Input schema for FirecrawlWebSearchTool."""

    title: str = Field(..., description="Title of the book.")

class FirecrawlWebSearchTool(BaseTool):
    name: str = "FirecrawlWebSearchTool"
    description: str = "Tool to retrieve Google results via Firecrawl."
    args_schema: Type[BaseModel] = FirecrawlWebSearchToolInput

    def _run(self, title: str) -> dict:
        url = "https://api.firecrawl.dev/v1/search"
        headers = {
            "Authorization": f"Bearer {os.getenv('FIRECRAWL_API_KEY')}",
            "Content-Type": "application/json",
        }
        payload = {"query": title, "limit": 5, "lang": "en"}

        response = requests.post(url, json=payload, headers=headers)

        return response.json()