# Copyright (C) Okahu Inc 2023-2024. All rights reserved

import os, sys
import chromadb
from typing import Any, Dict, List, Optional, Union
from llama_index.llms.nvidia_triton import NvidiaTriton
from llama_index.embeddings.openai import OpenAIEmbedding
import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.llms.nvidia_triton.utils import GrpcTritonClient
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import numpy as np
from okahu_apptrace.instrumentor import setup_okahu_telemetry
from okahu_apptrace.wrap_common import llm_wrapper
from okahu_apptrace.wrapper import WrapperMethod
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

class myTritonClient(GrpcTritonClient):
    def _generate_inputs(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        prompt: str,
        tokens: int = 300,
        temperature: float = 1.0,
        top_k: float = 1,
        top_p: float = 0,
        beam_width: int = 1,
        repetition_penalty: float = 1,
        length_penalty: float = 1.0,
        stream: bool = True,
    ) -> List[Union["grpcclient.InferInput", "httpclient.InferInput"]]:
        query = np.array(prompt, dtype=np.bytes_).reshape(1).astype(object)
        return [
            self._prepare_tensor("text_input", query),
        ]

def init(config_path:str):
    # setTritonEnvironmentVariablesFromConfig(config_path)
    setup_okahu_telemetry(
        workflow_name="llama_triton_wf",
        span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
        wrapper_methods=[
            WrapperMethod(
                package="llama_index.llms.nvidia_triton",
                object="NvidiaTriton",
                method="complete",
                span_name="llamaindex.triton",
                wrapper=llm_wrapper),
            ]
    )


def run(query):
    
    model_name = "flan_t5"
    triton_url: str = os.environ["TRITON_LLM_ENDPOINT"]
    triton_server = NvidiaTriton(server_url=triton_url, model_name=model_name, reuse_client=True)
    triton_server._client = myTritonClient(triton_url)


    # Creating a Chroma client
    # EphemeralClient operates purely in-memory, PersistentClient will also save to disk
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("quickstart")

    # construct vector store
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection,
    )
    dir_path = os.path.dirname(os.path.realpath(__file__))
    documents = SimpleDirectoryReader(dir_path + "/data").load_data()

    embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )

    query_engine = index.as_query_engine(llm=triton_server)
    response = query_engine.query("What did the author do growing up?")
    # resp = triton_server.complete("What is Latte?")
    print(response)


if __name__ == "__main__":
    init("config/config.ini")
    run("Good morning")

# {
#     "name": "llamaindex.retrieve",
#     "context": {
#         "trace_id": "0x66335f7fd712d6382734c08204a27734",
#         "span_id": "0x524ab11c3f52338f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0625e202c4182bcb",
#     "start_time": "2024-05-14T14:30:46.631916Z",
#     "end_time": "2024-05-14T14:30:47.286263Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "workflow_context_input": "What did the author do growing up?",
#         "workflow_context_output": "this is some sample text"
#     },
#     "events": [],
# }
# {
#     "name": "llamaindex.triton",
#     "context": {
#         "trace_id": "0x66335f7fd712d6382734c08204a27734",
#         "span_id": "0x9d95e6234b6455d9",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0625e202c4182bcb",
#     "start_time": "2024-05-14T14:30:47.290283Z",
#     "end_time": "2024-05-14T14:30:49.533267Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "openai_model_name": "flan_t5"
#     },
#     "events": []
# }
# {
#     "name": "llamaindex.query",
#     "context": {
#         "trace_id": "0x66335f7fd712d6382734c08204a27734",
#         "span_id": "0x0625e202c4182bcb",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": None,
#     "start_time": "2024-05-14T14:30:46.631569Z",
#     "end_time": "2024-05-14T14:30:49.533789Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "workflow_input": "What did the author do growing up?",
#         "workflow_name": "llama_triton_wf",
#         "workflow_output": "The context provided does not provide a clear answer to the question \"What did the author do growing up?\" as it does not provide a specific information about their background or activities.",
#         "workflow_type": "workflow.llamaindex"
#     },
#     "events": []
# }