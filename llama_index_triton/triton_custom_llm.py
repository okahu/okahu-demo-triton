import os, uuid
from typing import Any

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

from opentelemetry import trace
import requests

class TritonLlamaLLM(CustomLLM):
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "custom"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:

        triton_url: str = os.environ["TRITON_LLM_ENDPOINT"]

        ctx = trace.get_current_span().get_span_context()
        trace_id = '{trace:32x}'.format(trace=ctx.trace_id)
        span_id = '{span:16x}'.format(span=ctx.span_id)
        payload = {
            "id": '{trace_id}',
            "inputs": [
                {
                    "name": "text_input",
                    "datatype": "BYTES",
                    "shape": [1],
                    "data": [prompt],
                }
            ]
        }

        # eg. 00-80e1afed08e019fc1110464cfa66635c-00085853722dc6d2-00
        # The traceparent header uses the version-trace_id-parent_id-trace_flags format
        header_trace = {"traceparent": f"00-{trace_id}-{span_id}-00"}
        ret = requests.post(
            triton_url,
            json=payload,
            timeout=10,
            headers= header_trace,
        )

        res = ret.json()
        query_response = res["outputs"][0]["data"][0]

        return CompletionResponse(text=query_response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)