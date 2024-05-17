import os, sys
from typing import Any, Dict, List, Optional, Union
import logging
import chromadb
from llama_index.llms.nvidia_triton import NvidiaTriton
from credential_utilties.environment import setTritonEnvironmentVariablesFromConfig, setDataEnvironmentVariablesFromConfig, setOpenaiEnvironmentVariablesFromConfig
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.nvidia_triton.utils import GrpcTritonClient
from llama_index.llms.nvidia_triton.utils import _BaseTritonClient
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import numpy as np
from okahu_apptrace.instrumentor import setup_okahu_telemetry
from okahu_apptrace.wrapper import WrapperMethod,llm_wrapper
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings


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
    setDataEnvironmentVariablesFromConfig(config_path)
    setTritonEnvironmentVariablesFromConfig(config_path)

    logging.basicConfig()
    logging.getLogger().setLevel(logging.ERROR)
    setup_okahu_telemetry(
        workflow_name="llama_triton_wf4",
#        span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
#        wrapper_methods=[
#            WrapperMethod(
#                package="llama_index.llms.nvidia_triton",
#                object="NvidiaTriton",
#                method="complete",
#                span_name="tritonllm",
#                wrapper=llm_wrapper),
#            ]
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def setup_embedding(chroma_vector_store: ChromaVectorStore, embed_model):
    # Creating a Chroma client
    documents = SimpleDirectoryReader(input_files=
                             [os.environ["AZUREML_MODEL_DIR"] + "/coffee_llama_embedding/coffee.txt"]).load_data()

    storage_context = StorageContext.from_defaults(vector_store=chroma_vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )
    index.storage_context.persist(persist_dir=os.environ["AZUREML_MODEL_DIR"])

def get_vector_index() -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=os.environ["AZUREML_MODEL_DIR"] + "/coffee_llama_embedding")
    create_embedding = False
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    try:
        chroma_collection = chroma_client.get_collection("quickstart")
    except ValueError:
        chroma_collection = chroma_client.create_collection("quickstart")
        create_embedding = True
    # construct vector store
    chroma_vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection,
    )
    if create_embedding == True:
        setup_embedding(chroma_vector_store, embed_model)
    return VectorStoreIndex.from_vector_store(vector_store=chroma_vector_store, embed_model=embed_model)

def run():
    model_name = "flan_t5"
    triton_url: str = os.environ["TRITON_LLM_GRPC_ENDPOINT"]
    triton_server = NvidiaTriton(server_url=triton_url, model_name=model_name, reuse_client=True)
    triton_server._client = myTritonClient(triton_url)

    index = get_vector_index()
    query_engine = index.as_query_engine(llm=triton_server)

    while True:
        prompt = input("\nAsk a coffee question [Press return to exit]: ")
        if prompt == "":
            break
        response = query_engine.query(prompt)
        print(response)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage " + sys.argv[0] + " <config-file-path>")
        sys.exit(1)
    init(sys.argv[1])
    run()
