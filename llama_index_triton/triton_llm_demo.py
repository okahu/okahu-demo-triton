import os, sys
from typing import Any, Dict, List, Optional, Union
import logging
import chromadb
from credential_utilties.environment import setTritonEnvironmentVariablesFromConfig, setDataEnvironmentVariablesFromConfig
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from triton_custom_llm import TritonLlamaLLM

def init(config_path:str):
    setDataEnvironmentVariablesFromConfig(config_path)
    setTritonEnvironmentVariablesFromConfig(config_path)

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
    triton_server = TritonLlamaLLM()
    index = get_vector_index()
    query_engine = index.as_query_engine(llm=triton_server)

    while True:
        prompt = input("\nAsk a coffee question [Press return to exit]: ")
        if prompt == "":
            break
        response = query_engine.query(prompt)
        print(response)

def main():
    init(sys.argv[1])
    run()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage " + sys.argv[0] + " <config-file-path>")
        sys.exit(1)
    main()