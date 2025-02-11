from llama_index.llms.openai import OpenAI
import os
from menu_mapping.helper_classes.utility import MenuMappingUtility
import json
from llama_index.llms.gemini import Gemini
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai_like import OpenAILike


class LLMHelper:
    def __init__(self, model, temperature=0.3):
        if "deepseek" in model:
            # self.llm = DeepSeek(model=self.model, api_key=os.getenv('DEEP_SEEK_API_KEY'))
            self.llm = OpenAILike(model="deepseek-chat", api_base="https://api.deepseek.com/v1", api_key=os.getenv('DEEP_SEEK_API_KEY'), is_chat_model=True)
        elif "claude" in model:
            self.llm = Anthropic(model=model, api_key=os.getenv('CLAUDE_API_KEY'))
        elif "gemini" in model:
            self.llm = Gemini(model=model, api_key=os.getenv('GEMINI_API_KEY'))
        else:
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
                    Convert 'dosai' to 'dosa'. 'Gateaux' to 'cake', 'Poori' to 'Puri', 'Pattice' to 'Patties'

                    You need to determine whether an item name is ambiguous or specific. A food item is considered ambiguous if its name is vague and not a specific food, such as when it only states unspecific descriptors.
                    Given are precious examples/ rules that will help you determine if an item is ambiguous or not.
                    Ambiguous Items:
                    - Vague descriptors: 'Chocolate Desire Heart', 'juice of the day', 'Paneer Snacks', 'full non veg chicken meal', 'Signature Non Veg Sandwich'
                    - Generic meal terms: 'Dinner', 'Variety', 'Menu', 'Meal', 'Lunch', 'Chapati Non Veg', 'Non Veg Rice'
                    - Contains 'Thali': any item with 'thali' in its name.
                    - Single flavour names: 'chocolate', 'mango', 'strawberry', 'butterscotch', 'Blackcurrent'.
                    - Non sepcific descriptors: 'sweet', 'sabji', 'Gravy', 'Veg Dish', 'Non Veg combo'.
                    Non-Ambiguous Items:
                    - Detailed items: 'desire heart cake', 'Juice', 'chapati', 'Paratha', 'Rice', 'Veg Nuggets', 'Sweet Lassi', 'Hi Tea', 'Choco Lava'
                    - Combo/dish names: 'idli vada combo', 'Veg Manchurian Noodle Combo', 'basanti dum pulao mutton kosha combo', 'Kaju Katli Sweet', 'aloo sabji', 'rice bowl', 'date and walnut cake'.
                    - Descriptive beverages/desserts: 'dry fruit milkshake', 'Mango Masti', 'Chocolate Drink', 'Hot Chocolate', 'Butter Scotch Ice Cream Shake', 'Death By Chocolate'.
                    - Single fruit/vegetable names: (e.g., 'apple', 'banana' etc. are specific).
                    - Common food items: 'Tea', 'Biscuit'

                    Also tell if the item is a retail store food item (mrp) or a restaurant dish (non_mrp).
                    MRP/Non-MRP Classification:
                    - Non-MRP (Restaurant Dish): Items like 'samosa', 'pakora', 'muffin', 'bread', 'Veg Sandwich', 'curd', 'Lemon Juice', 'Kaju Katli', 'Sunflower Seeds Roasted', 'Belgian Milk Chocolate', 'Naughty Nutella', 'Choco Truffle'
                    as well as all fruits/fruit juices and vegetables/vegetable dishes.
                    - Note: 'Maggie', 'Horlicks', 'Boost', 'Bournvita', 'Mr Chicken Tikka Burger', 'Pesto Paneer Puff' items are Non-MRP items.
                    - MRP (Retail Store): Packaged or branded items not covered above. Like 'Amul xyz', 'Nestle xyz' and so on.

                    Formatting & Splitting Rules:
                    - Capitalization: Each important word must start with a capital letter (e.g., 'Chicken Egg Biryani').
                    - Split Multi-Component Items: e.g., 'Chapati 3 Egg Curry' should become 'Chapati | Egg Curry'
                    'Aloo Paratha - 1 No With Channa Masala - 60 Grm Curd' should become 'Aloo Paratha | Channa Masala | Curd'.
                    - Single Items: 'Chicken Egg Biryani', 'Chicken Paratha', 'Chicken Egg Roll', 'oats apple jar', 'Palak Onion Pakoda', 'Strawberry Limeade', 
                    'bread jam toast', 'bread butter jam', 'butter toast'are treated as one item. You get the idea.
                    - Remove any occurrence of 'add on' or 'addon'.
                    - Preserve Brand Names: Do not remove brand names like 'Amul' or 'Domino's'.
                    - Final Output: JSON format with double quotes enclosed in ```json { }``` and the 'name' field must not contain any commas


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
                    }"""
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
