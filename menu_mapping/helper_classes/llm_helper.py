import openai
from llama_index.llms.openai import OpenAI
import os


class LLMHelper:
    def __init__(self, model):
        openai.api_key = os.getenv('OPEN_API_KEY')
        self.llm = OpenAI(model=model, temperature=0.3)

    def execute(self, prompt):
        response = self.llm.complete(prompt)
        return response.text
