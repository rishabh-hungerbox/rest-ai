import openai
import json
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext, load_index_from_storage
import os
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
from jsonschema import validate
from llama_index.core.node_parser import SentenceWindowNodeParser
import csv
import random
from datetime import datetime
import unicodedata
import re
from menu_mapping.helper_classes.llm_helper import LLMHelper


class MenuMapperAI:
    def __init__(self, prompt_id, model, embedding, similarity_top_k, benchmark_on, debug_mode, sampling_size):
        self.prompt_id = prompt_id
        self.model = model
        self.embedding = embedding
        self.similarity_top_k = similarity_top_k
        self.benchmark_on = benchmark_on
        self.global_index = None
        self.debug_mode = debug_mode
        self.app_id = f"prompt_{self.prompt_id}_{self.model}_{self.embedding}_top_k_{self.similarity_top_k}"
        self.sampling_size = sampling_size
        self.tru = None
        self.tru_recorder = None

        load_dotenv()
        openai.api_key = os.getenv('OPEN_API_KEY')
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        documents, self.item_id_map = self.fetch_data()
        self.global_index = self.generate_vector_index(documents)
        self.prompt = self.fetch_prompt()

    def execute(self, child_menu_name):
        return self.generate_response_debug(child_menu_name)

    @staticmethod
    def normalize_string(s: str) -> str:
        s = s.lower()
        s = unicodedata.normalize('NFKC', s)
        s = re.sub(r"[^a-zA-Z0-9]", " ", s)  # Replace non-alphanumeric characters with space
        s = " ".join(s.split())  # Remove extra whitespace
        return s
    
    def process_response(self, response, item_id_map, threshold=0.6) -> list:
        relevant_items = json.loads(str(response).strip("```json").strip("```"))
        filtered_items = []
        for item in relevant_items:
            if item['relevance_score'] >= threshold:
                if item['id'] in item_id_map:
                    item_data = item_id_map[item['id']]
                    filtered_items.append({
                        "id": item_data['id'],
                        "name": item_data['name'],
                        "usage": item_data['usage'],
                        "relevance_score": item['relevance_score']
                    })
                else:
                    print(f"Item {item['id']} not found in item_id_map, model halucinated!!!!!")
                    return [{
                        "id": -1,
                        "name": item['name'],
                        "usage": 0,
                        "relevance_score": item['relevance_score']
                    }]

        sorted_items = sorted(
                filtered_items,
                key=lambda x: (x['relevance_score'], x['usage']),
                reverse=True
            )
        return sorted_items

    def fetch_data(self):
        # JSON Schema
        json_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "description": "Schema for a list of items with id, name, and usage",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "description": "Unique identifier for the item",
                        "type": "integer"
                    },
                    "name": {
                        "description": "Name of the item",
                        "type": "string"
                    },
                    "usage": {
                        "description": "Usage count of the item",
                        "type": "integer",
                        "minimum": 0
                    }
                },
                "required": ["id", "name", "usage"]
            }
        }

        file_path = "menu_mapping/input/large_sku_with_usage.json"
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
            validate(instance=data, schema=json_schema)
        except Exception as e:
            print(f"Error: {str(e)}")
            return

        item_id_map = {}

        # Convert each row to a Document
        documents = [Document(text="ID,NAME")]
        for item in data:
            normalized_item_name = MenuMapperAI.normalize_string(item['name'])
            if ('bulk' in normalized_item_name or 'inv-' in normalized_item_name or 'test' in normalized_item_name or len(normalized_item_name) < 4):
                continue
            documents.append(Document(text=f"{item['id']},{normalized_item_name}"))
            item_id_map[item['id']] = {
                "id": item['id'],
                "name": item['name'],
                "usage": item['usage']
            }

        # saving file for testing purpose, to be removed
        documents_path = os.path.join(os.path.dirname(file_path), "processed_documents.txt")
        with open(documents_path, "w") as file:
            for doc in documents:
                file.write(doc.text + "\n")

        return documents, item_id_map

    def generate_vector_index(self, documents):
        node_parser = SentenceWindowNodeParser.from_defaults(
                window_size=3,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            )

        # llm settings
        llm = OpenAI(model=self.model, temperature=0.3)
        Settings.llm = llm
        Settings.embed_model = OpenAIEmbedding(model=self.embedding)
        Settings.node_parser = node_parser

        # Build the index
        if not os.path.exists("./menu_mapping_index"):
            print("Creating index...")
            index = VectorStoreIndex.from_documents(documents=documents)
            index.storage_context.persist(persist_dir="./menu_mapping_index")
            print("Index created successfully!")
        else:
            print("Loading pre-existing index...")
            index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir="./menu_mapping_index"))
            print("Pre-existing Index loaded!")

        return index

    def fetch_prompt(self):
        try:
            with open("menu_mapping/input/prompt_data.csv", 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if int(row['prompt_id']) == self.prompt_id:
                        return row['prompt']
            raise ValueError(f"No prompt found for prompt_id: {self.prompt_id}")

        except FileNotFoundError:
            raise FileNotFoundError("Prompt file 'xyz.csv' not found")
        except Exception as e:
            raise Exception(f"Error reading prompt file: {str(e)}")

    def read_and_sample_csv(self, file_path):
        input_data = {}
        try:
            with open(file_path, 'r') as file:
                # Read all rows
                reader = csv.DictReader(file)
                rows = list(reader)

                # Select random rows
                sample_size = min(self.sampling_size, len(rows))
                sampled_rows = random.sample(rows, sample_size)

                # Print id and name for each sampled row
                for row in sampled_rows:
                    normalized_item_name = MenuMapperAI.normalize_string(row['name'])
                    input_data[row['id']] = {
                        "id": row['id'],
                        "name": normalized_item_name,
                        "mv_id": row['mv_id'],
                        "mv_name": row['mv_name']
                    }
                    print(f"ID: {row['id']}, Name: {normalized_item_name}")

        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found")
        except Exception as e:
            print(f"Error: {str(e)}")
        return input_data

    def generate_response_debug(self, user_input):
        retriever = self.global_index.as_retriever(similarity_top_k=self.similarity_top_k)
        item_name = ItemSpellCorrector('gpt-4o-mini').correct_item_spelling(user_input)
        output_text = f'{user_input},{item_name},"'
        nodes = retriever.retrieve(item_name)
        print()
        print('Item Name: ', item_name)
        text = "ID,Food Item Name,Vector Score\n"
        for node in nodes:
            text += f"{node.node.text},{node.score}\n"
            output_text += f'{node.node.text},{node.score}\n'
            print(node.node.text, node.score)
        output_text += '",'

        # preparing query engine on the filtered index
        filtered_index = VectorStoreIndex.from_documents([Document(text=text)])
        query_engine = filtered_index.as_query_engine(embeddings_enabled=True)

        response = query_engine.query(self.prompt + user_input)
        print("response: ", response)
        relevant_items = self.process_response(response, self.item_id_map)
        output_text += f'"{json.dumps(relevant_items).replace("\"", "\"\"")}"\n'

        self.save_output(output_text)

        return relevant_items
    
    def save_output(self, output_text):
        file_path = f"menu_mapping/output/output_{datetime.now().strftime("%Y-%m-%d")}.csv"

        if not os.path.exists(file_path):
            with open(file_path, "w") as file:
                file.write("User Input, Spell Correction, Vector Tokens, Response\n")

        with open(file_path, "a") as file:
            file.write(output_text)
        


class ItemSpellCorrector:
    def __init__(self, model):
        self.model = model

    def correct_item_spelling(self, item_name):
        prompt = """correct the spelling of this Indian food item, "%{item_name}"
                    reply with the exact answer only
                    keep definitive spellings for indian food items like 'bhaji', 'chapati', 'paratha' and so on
                    Note: Things like 'parotta' should not get converted to 'paratha'
                    """
        name = LLMHelper(self.model).execute(prompt.format(item_name=item_name))
        return MenuMapperAI.normalize_string(name)


ai = MenuMapperAI(prompt_id=4, model="gpt-4o", embedding="text-embedding-3-small", similarity_top_k=10, benchmark_on=False, debug_mode=False, sampling_size=50)


def get_master_menu_response(child_menu_name: str):
    return ai.execute(child_menu_name)
