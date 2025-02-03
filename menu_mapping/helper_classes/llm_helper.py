import openai
from llama_index.llms.openai import OpenAI
import os
from menu_mapping.helper_classes.utility import MenuMappingUtility


class LLMHelper:
    def __init__(self, model, temperature=0.3):
        openai.api_key = os.getenv('OPEN_API_KEY')
        self.llm = OpenAI(model=model, temperature=temperature)

    def execute(self, prompt):
        response = self.llm.complete(prompt)
        return response.text


class ItemSpellCorrector:
    def __init__(self, model):
        self.model = model

    def correct_item_spelling(self, item_name):
        prompt = """correct the spelling of this Indian/retail food item, "{item_name}"
                    reply with the exact answer only
                    keep definitive spellings for indian food items like 'pakora', 'bhaji', 'chapati', 'paratha' and so on
                    Note: Things like 'parotta' should not get converted to 'paratha'
                    Note: Possible items can also be brands like 'perk', 'lays', 'boost' etc
                    """
        name = LLMHelper(self.model).execute(prompt.format(item_name=item_name))
        return MenuMappingUtility.normalize_string(name)


class Evaluator:
    def __init__(self, model):
        self.model = model

    def item_evaluator(self, predicted_item, user_item) -> bool:
        prompt = prompt = f"""Is "{predicted_item}" a valid match for "{user_item}"? Consider:
                            - Ingredients
                            - Cooking style
                            - Regional/cultural context
                            Answer ONLY 'YES' or 'NO'.
                            """
        answer = LLMHelper(self.model, temperature=0).execute(prompt)
        return answer
