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
        file_path = "menu_mapping/input/root_items_input.csv"
        try:
            with open(file_path, "r") as file:
                reader = csv.DictReader(file)
                rows = list(reader)

        except Exception as e:
            print(f"Error: {str(e)}")
            return

        item_id_map = {}

        # Convert each row to a Document
        documents = [Document(text="ID,NAME")]
        for row in rows:
            documents.append(Document(text=f"{row['id']},{row['name']}"))
            item_id_map[str(row['id'])] = {
                "id": row['id'],
                "name": row['name'],
                "trace_ids": row['trace_ids']
            }

        # saving file for testing purpose, to be removed
        documents_path = os.path.join(os.path.dirname(file_path), "processed_documents_root.txt")
        with open(documents_path, "w") as file:
            for doc in documents:
                file.write(doc.text + "\n")

        return documents, item_id_map
