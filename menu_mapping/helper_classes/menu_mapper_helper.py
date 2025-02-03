import openai
import json
from llama_index.llms.openai import OpenAI
from menu_mapping.helper_classes.llm_helper import LLMHelper, ItemSpellCorrector, Evaluator
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext, load_index_from_storage
import os
from datetime import datetime
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import QueryBundle
from llama_index.embeddings.openai import OpenAIEmbedding
from trulens.core import TruSession
from dotenv import load_dotenv
from menu_mapping.helper_classes.tru_lens_helper import TruLensHelper
from llama_index.core.node_parser import SentenceWindowNodeParser
import csv
from llama_index.core.postprocessor import LLMRerank
from menu_mapping.helper_classes.utility import MenuMappingUtility
from llama_index.llms.anthropic import Anthropic


class MenuMapperAI:
    def __init__(self, prompt_id, model, embedding, similarity_top_k, benchmark_on, debug_mode, sampling_size, with_reranker):
        self.prompt_id = prompt_id
        self.model = model
        self.embedding = embedding
        self.similarity_top_k = similarity_top_k
        self.benchmark_on = benchmark_on
        self.global_index = None
        self.debug_mode = debug_mode
        self.with_reranker = with_reranker
        self.app_id = f"prompt_{self.prompt_id}_{self.model}_{self.embedding}_top_k_{self.similarity_top_k}_with_reranker_{self.with_reranker}"
        self.sampling_size = sampling_size
        self.tru = None
        self.tru_recorder = None
        self.llm = None

        load_dotenv()
        openai.api_key = os.getenv('OPEN_API_KEY')
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        documents, self.item_id_map = MenuMappingUtility.fetch_data()
        self.global_index = self.generate_vector_index(documents)
        self.prompt = self.fetch_prompt()

    def execute(self, child_menu_name):
        return self.generate_response_debug(child_menu_name)

    # def execute(self):
    #     input_data = MenuMappingUtility.read_and_sample_csv("menu_mapping/input/input_data.csv")
    #     documents, item_id_map = MenuMappingUtility.fetch_data()
    #     self.global_index = self.generate_vector_index(documents)
    #     self.retriever = self.global_index.as_retriever(similarity_top_k=self.similarity_top_k)
    #     prompt = self.fetch_prompt()

    #     if self.benchmark_on:
    #         print("inializing trulens...")
    #         self.tru, self.tru_recorder = self.get_trulens(self.global_index.as_query_engine())
    #         print("trulens inialized!")

    #     if not self.debug_mode:
    #         input_data = MenuMappingUtility.read_and_sample_csv("menu_mapping/input/input_data.csv")
    #         self.generate_response(prompt, item_id_map, input_data)
    #     else:
    #         self.generate_response_debug(prompt, item_id_map)

    def process_response(self, response, item_id_map, threshold=0.65) -> list:
        try:
            relevant_items = json.loads(str(response).strip("```json").strip("```"))
        except Exception as e:
            print(f"Error processing response: {e}")
            return [{
                        "id": -1,
                        "name": 'LLM JSON parsing failed',
                        "usage": 0,
                        "relevance_score": 0
                    }]
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

    def generate_vector_index(self, documents):
        node_parser = SentenceWindowNodeParser.from_defaults(
                window_size=3,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            )

        # llm settings
        if "deepseek" in self.model:
            self.llm = OpenAILike(model="deepseek-chat", api_base="https://api.deepseek.com/v1", api_key=os.getenv('DEEP_SEEK_API_KEY'), is_chat_model=True)
        elif "claude" in self.model:
            self.llm = Anthropic(model=self.model, api_key=os.getenv('CLAUDE_API_KEY'))
        else:
            self.llm = OpenAI(model=self.model, temperature=0.3)
        Settings.llm = self.llm
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

    def get_trulens(self, query_engine):
        tru = TruSession()
        tru_recorder = TruLensHelper.get_prebuilt_trulens_recorder(query_engine, self.app_id)
        return tru, tru_recorder

    def generate_response(self, prompt, item_id_map, input_data):
        RETRY_COUNT = 3
        ouput_text = 'ID,Name,Cleaned Name,Master Menu ID,Master Menu Name,Eval Input,Predicted Id,Predicted Name,Eval Prediction, Score, Vector Search Best,Reranker Best\n'

        dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        file_path = f"menu_mapping/output/output_mapper_{self.model}_{dt}.csv"
        with open(file_path, "w") as file:
            file.write(ouput_text)

        for data in input_data.values():
            item = data['name']

            item_name = ItemSpellCorrector('gpt-4o-mini').correct_item_spelling(item)
            print(f'Item Name: {item}, Correct Name: {item_name}')
            nodes, vs_best, vs_score, llmre_best, llmre_score = self.get_filtered_nodes(item_name)
            if len(nodes) == 0:
                print("Reranker returned nothing !!!!!")
                eval_input = Evaluator('gpt-4o-mini').item_evaluator(data['mv_name'], item_name)
                ouput_text += f"{data['id']},{data['name']},{data['mv_id']},{data['mv_name']},'{eval_input}','NOT FOUND','NOT FOUND','NULL','0','NULL','NULL'\n"
                continue

            text = "ID,Food Item Name,Vector Score\n"
            for node in nodes:
                text += f"{node.node.text},{node.score}\n"

            # preparing query engine on the filtered index
            filtered_index = VectorStoreIndex.from_documents([Document(text=text)])
            query_engine = filtered_index.as_query_engine(embeddings_enabled=True)

            if self.benchmark_on:
                with self.tru_recorder:
                    response = query_engine.query(prompt + item_name)
            else:
                while RETRY_COUNT > 0:
                    try:
                        response = query_engine.query(prompt + item_name)
                        break
                    except Exception as e:
                        print(f"Error querying: {e}")
                        RETRY_COUNT -= 1
            if not response:
                raise Exception("LLM returned nothing even after retrying")
            print("response: ", response)
            relevant_items = self.process_response(response, item_id_map)

            print(f"Child Menu Name: {item_name}\nRelevant Items:\n{json.dumps(relevant_items, indent=4)}\n")
            print("Most Relevant Item:\n")
            selected_id, selected_name, eval_prediction, eval_input, llm_score = 'NOT FOUND', 'NOT FOUND', 'NULL', 'NULL', 0
            if len(relevant_items) > 0:
                selected_id = relevant_items[0]['id']
                selected_name = relevant_items[0]['name']
                llm_score = relevant_items[0]['relevance_score']
                eval_prediction = Evaluator('gpt-4o-mini').item_evaluator(selected_name, item_name)
                eval_input = Evaluator('gpt-4o-mini').item_evaluator(data['mv_name'], item_name)
                print(json.dumps(relevant_items[0], indent=4))
            else:
                eval_input = Evaluator('gpt-4o-mini').item_evaluator(data['mv_name'], item_name)
                print("None")
            ouput_text = f"""{data['id']},{data['name']},{item_name},{data['mv_id']},{data['mv_name']},{eval_input},{selected_id},{selected_name},{eval_prediction},{llm_score}, ""{vs_best}"",""{llmre_best}""\n"""
            with open(file_path, "a") as file:
                file.write(ouput_text)

        return

    def generate_response_debug(self, prompt, item_id_map):
        while True:
            user_input = input("Enter an item name to search (or 'quit' to exit): ")
            if user_input == 'quit':
                break
            self.prompt_id = int(input("Prompt ID: "))
            prompt = self.fetch_prompt()

            item_name = ItemSpellCorrector('gpt-4o-mini').correct_item_spelling(user_input)
            print(f'Item Name: {user_input}, Correct Name: {item_name}')
            nodes, _, _, _, _ = self.get_filtered_nodes(item_name)
            if len(nodes) == 0:
                print("Reranker returned nothing !!!!!")
                continue

            text = "ID,Food Item Name,Vector Score\n"
            for node in nodes:
                text += f"{node.node.text},{node.score}\n"
            # preparing query engine on the filtered index
            filtered_index = VectorStoreIndex.from_documents([Document(text=text)])
            query_engine = filtered_index.as_query_engine(embeddings_enabled=True)

            if self.benchmark_on:
                with self.tru_recorder:
                    response = query_engine.query(prompt + item_name)
            else:
                response = query_engine.query(prompt + item_name)

            print("prompt: ", prompt + item_name)
            print("response: ", response)
            relevant_items = self.process_response(response, item_id_map)

            print(f"Child Menu Name: {user_input}\nRelevant Items:\n{json.dumps(relevant_items, indent=4)}\n")
            print("Most Relevant Item:\n")
            if len(relevant_items) > 0:
                print(json.dumps(relevant_items[0], indent=4))
            else:
                print("None")

        return

    def get_filtered_nodes(self, item_name):
        vs_best, vs_score, llmre_best, llmre_score = 'NULL', 0, 'NULL', 0
        if self.with_reranker:
            self.similarity_top_k = 15
        self.retriever = self.global_index.as_retriever(similarity_top_k=self.similarity_top_k)
        nodes = self.retriever.retrieve(item_name)
        if len(nodes) > 0:
            vs_best = nodes[0].node.text
            vs_score = nodes[0].score
        if self.with_reranker:
            reranker = LLMRerank(
                top_n=self.similarity_top_k,
                llm=self.llm
            )
            nodes = reranker.postprocess_nodes(
                nodes, QueryBundle(f"Food items like {item_name}. Seperate item from adjectives and quantities.")
            )
            if len(nodes) > 0:
                llmre_best = nodes[0].node.text
                llmre_score = nodes[0].score

        print("ID,Food Item Name,Vector Score")
        for node in nodes:
            print(f"- {node.node.text}, {node.score}")
        return nodes, vs_best, vs_score, llmre_best, llmre_score


ai = MenuMapperAI(prompt_id=4, model="gpt-4o-mini", embedding="text-embedding-3-small", similarity_top_k=10, benchmark_on=False, debug_mode=False, sampling_size=50,with_reranker=True)


def get_master_menu_response(child_menu_name: str):
    return ai.execute(child_menu_name)
