import openai
from llama_index.llms.openai import OpenAI
import os
from menu_mapping.helper_classes.utility import MenuMappingUtility
import json


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
    

class ItemFormatter:
    def __init__(self, model):
        self.model = model

    def format(self, item_name):
        prompt = """Given the name of a food item name (can be Indian), extract the different food items present in the name and reply with a ' | ' seperated string (Look at the example). Remove any sort of price or quantity from the said food item name.
                    Keep definitive spellings for indian food items like 'pakora', 'bhaji', 'chapati', 'paratha', 'idli', 'bhatura' and so on. Example correct 'Chapathi', 'Chappathi' to 'Chapati'. Correct 'Laccha' to 'Lachha'.
                    Spell correct 'rool' to 'roll', 'Subji' to 'sabji' and so on.
                    Note: Things like 'parotta' should not get converted to 'paratha'.
                    Also get rid of unnecessary words like 'special' or places name like 'kerala' for definitive items while spell correcting like 'life tea' to 'tea', 'kerala tea' to 'tea'.
                    Tell if the item is ambigious in name or not. Ambiguity arises with unspecific names like 'juice of the day', while specific names like 'Juice' are not ambiguous.
                    'Combo', 'Dinner', 'variety', 'Menu', 'Thali', 'Meal' and 'Lunch' may be considered as ambiguous as they don't specify any item. But items like 'idli vada combo' are not ambiguous as they state the items.
                    Things like just 'sweet' are also ambiguous. But 'Kaju Katli Sweet' is not ambiguous as it is a specific item.
                    Also tell if the item is a retail store food item (mrp) or a restaurant dish (non_mrp).
                    Note: items like 'samosa', 'pakora', 'muffin', 'bread', 'Veg sandwich', 'curd' , 'Lemon Juice' and so on are not MRP items.
                    Formatted Name should have proper capitalization (start of each important word in capital) (look at the example below). 
                    Note : 'Chicken Egg Biryani', 'Chicken paratha', 'Chicken Egg roll' are same single items while 'Chapati 3 Egg curry' should be separated like 'Chapati, Egg Curry'. 'Aloo Paratha - 1 No With Channa Masala - 60 Grm Curd' Should be separated like 'Aloo Paratha, Channa Masala, Curd'
                    Items like 'fried rice chicken' and 'chicken fried rice' are same single items.
                    Note: Brand names should are important and should not be removed like 'Amul', 'Domino's' etc.
                    Output should be in json format with double quotes and enclosed in ```json {} ```
                    Output 'name' field should not contain any commas. Remove 'add on'/'addon' from input if present.

                    Example:
                    Input: '2 piece dossa & idlly 50 milligram 30 /- Rs'
                    Output:```json{
                    "name": "Dosa | Idli",
                    "ambiguous": 0,
                    "is_mrp": 0
                    }```

                    Input: 'glazed night snack'
                    Output:```json{
                    "name": "glazed night snack",
                    "ambiguous": 1,
                    "is_mrp": 0
                    }```

                    food item is """
        response = LLMHelper(self.model).execute(f'{prompt}{item_name}')
        try:
            response = str(response).replace("'", '"')
            formated_item = json.loads(response.strip("```json").strip("```"))
        except Exception as e:
            print(f"Error processing response: {e}")
            return {
                        "name": 'LLM JSON parsing failed',
                        "ambiguous": 1,
                        "is_mrp": 0
                    }
        
        return formated_item


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
