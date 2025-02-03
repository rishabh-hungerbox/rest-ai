import numpy as np
from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI
)


class TruLensHelper:

    @staticmethod
    def get_prebuilt_trulens_recorder(query_engine, app_id):
        openai = OpenAI(api_key="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")
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
