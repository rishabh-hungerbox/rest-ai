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
        prompt = """Given the name of a food item name (can be Indian), extract the different food items present in the name and reply with a ' | ' seperated string (Look at the example). Remove any sort of price or quantity from the said food item name for the name column.
                    Example: 'Medium Pizza Non veg' becomes 'Pizza Non Veg', 'Sambar Idli' becomes 'Sambar | Idli', 'Veg Manchurian Noodle Combo' becomes 'Veg Manchurian | Noodles' and so on.
                    Also extract the quantity (in brackets) and price (in Rs) from the said food item name and reply with a ' | ' seperated string for the quantity_details column.
                    Keep definitive spellings for indian food items like 'pakora', 'bhaji', 'bajji', 'chapati', 'paratha', 'idli', 'bhatura' and so on. Example correct 'Chapathi', 'Chappathi' to 'Chapati'. Correct 'Laccha' to 'Lachha'.
                    Don't get confused with 'bajji' and 'bajji' as both are different items.
                    Spell correct 'rool' to 'roll', 'Subji' to 'sabji' and so on.
                    Note: Things like 'parotta' should not get converted to 'paratha' and 'Kal Dosa' should remain as 'Kal Dosa'.
                    Also get rid of unnecessary words like 'special', 'spl', 'mini', 'jumbo', 'large', 'medium', 'small' in the item name. Do not remove important things like 'non veg' or 'veg'
                    Convert 'dosai' to 'dosa'.

                    You need to determine whether an item name is ambiguous or specific. A food item is considered ambiguous if its name is vague and not a specific food, such as when it only states unspecific descriptors.
                    For example, names like 'Chocolate Desire Heart', 'juice of the day' 'Paneer Snacks', 'full non veg chicken meal' are ambiguous food items. As in snacks we can hav any item.
                    Generic terms such as 'Dinner', 'Variety', 'Menu', 'Thali', 'Meal', and 'Lunch' are ambiguous because they don't specify any particular item. (Note: any item with 'thali' in its name is automatically considered ambiguous.) 
                    
                    A single word like 'sweet' or 'sabji' is ambiguous. On the other hand, an item is not ambiguous if it provides specific details about the item. 
                    For instance, 'desire heart cake' or 'Juice' are considered specific.
                    Not Ambigious Items -> 'idli vada combo', 'Veg Manchurian Noodle Combo' 'basanti dum pulao mutton kosha combo', 'Kaju Katli Sweet', 'aloo sabji', 'rice bowl', 'date and walnut cake'.
                    Reason -> They clearly indicate what the item is and therefore are not ambiguous. 
                    Note: 'dry fruit milkshake', 'Mango Masti', 'Chocolate Drink', 'Hot Chocolate', 'Butter Scotch Ice Cream Shake', 'Death By Chocolate' are specific enough for this usecase and thus are not ambiguous.
                    Note: single fruit and vegetable names are not ambiguous and are specific for this usecase.
                    Note: Even though 'Tea', 'Biscuit' is a general term, it should be treated as a specific item, not an ambiguous one."

                    Also tell if the item is a retail store food item (mrp) or a restaurant dish (non_mrp).
                    Note: items like 'samosa', 'pakora', 'muffin', 'bread', 'Veg sandwich', 'curd' , 'Lemon Juice' and so on are not MRP items and are not ambiguous.
                    Formatted Name should have proper capitalization (start of each important word in capital) (look at the example below).
                    Note : 'Chicken Egg Biryani', 'Chicken paratha', 'Chicken Egg roll' are same single items while 'Chapati 3 Egg curry' should be separated like 'Chapati, Egg Curry'. 'Aloo Paratha - 1 No With Channa Masala - 60 Grm Curd' Should be separated like 'Aloo Paratha, Channa Masala, Curd'
                    Items like 'fried rice chicken', 'usal pav', 'bread butter jam', 'oats apple jar', 'palak onion pakoda'and 'chicken fried rice' are single items.
                    Note: Brand names should are important and should not be removed like 'Amul', 'Domino's' etc.
                    Output should be in json format with double quotes and enclosed in ```json {} ```
                    Output 'name' field should not contain any commas. Remove 'add on'/'addon' from input if present.

                    Example:
                    Input: '2 piece dossa & idlly 50 milligram 30 /- Rs'
                    Output:```json{
                    "name": "Dosa | Idli",
                    "quantity_details": "Dosa (2 piece) | Idli (50 mg) [30 Rs]",
                    "ambiguous": 0,
                    "is_mrp": 0
                    }```

                    Input: 'glazed night snack'
                    Output:```json{
                    "name": "glazed night snack",
                    "quantity_details": "glazed night snack",
                    "ambiguous": 1,
                    "is_mrp": 0
                    }```
                    
                    Input: 'veg manchurian noodle combo
                    Output:```json{
                    "name": "Veg Manchurian | Noodles",
                    "quantity_details": "Veg Manchurian | Noodles",
                    "ambiguous": 0,
                    "is_mrp": 0
                    }```
                    
                    Input: '1 litre of milk 50 -/'
                    Output:```json{
                    "name": "Milk",
                    "quantity_details": "Milk (1 l) [50 Rs]",
                    "ambiguous": 0,
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
