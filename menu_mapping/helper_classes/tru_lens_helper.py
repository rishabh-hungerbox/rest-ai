import numpy as np
from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI
)
import os


class TruLensHelper:

    @staticmethod
    def get_prebuilt_trulens_recorder(query_engine, app_id):
        openai = OpenAI(api_key=os.getenv('OPEN_API_KEY'))
        groundedness = (
            Feedback(openai.groundedness_measure_with_cot_reasons, name="Groundedness")
            .on(TruLlama.select_source_nodes().node.text)
            .on_output()
        )
        qa_relevance = (
            Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
            .on_input_output()
        )
        qs_relevance = (
            Feedback(openai.relevance_with_cot_reasons, name="Context Relevance")
            .on_input()
            .on(TruLlama.select_source_nodes().node.text)
            .aggregate(np.mean)
        )
        feedbacks = [qa_relevance, qs_relevance, groundedness]
        tru_recorder = TruLlama(
            query_engine,
            app_id="Menu Mapper",
            app_version=app_id,
            feedbacks=feedbacks
            )
        return tru_recorder
