import os
import csv
import unicodedata
import re
from llama_index.core import Document
import json
from jsonschema import validate


class MenuMappingUtility:

    @staticmethod
    def read_and_sample_csv(file_path):
        input_data = {}
        try:
            with open(file_path, 'r') as file:
                # Read all rows
                reader = csv.DictReader(file)
                rows = list(reader)
                sorted_rows = sorted(rows, key=lambda row: int(row['id']))

                # Select random rows
                # sample_size = min(self.sampling_size, len(rows))
                # sampled_rows = random.sample(rows, sample_size)

                # Print id and name for each sampled row
                # for row in sampled_rows:
                for row in sorted_rows:
                    if int(row['id']) <= 720465:
                        continue
                    normalized_item_name = MenuMappingUtility.normalize_string(row['name'])
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

    @staticmethod
    def normalize_string(s: str) -> str:
        s = s.lower()
        s = unicodedata.normalize('NFKC', s)
        s = re.sub(r"\baddon\b", "", s)  # Remove the exact keyword 'addon'
        s = re.sub(r"\b\d+\s*ml\b", "", s)  # Matches numbers followed by 'ml' with optional spaces
        s = re.sub(r"[^a-zA-Z0-9]", " ", s)  # Replace non-alphanumeric characters with space
        s = " ".join(s.split())  # Remove extra whitespace
        return s

    @staticmethod
    def fetch_data():

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
            normalized_item_name = MenuMappingUtility.normalize_string(item['name'])
            if ('bulk' in normalized_item_name or 'inv-' in normalized_item_name or 'test' in normalized_item_name or len(normalized_item_name) < 3):
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
