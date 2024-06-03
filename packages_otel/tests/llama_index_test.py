# Copyright (C) Okahu Inc 2023-2024. All rights reserved

import json
import logging
import os.path
import time
from typing import List
import unittest
from unittest.mock import ANY, patch
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
import requests
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

from okahu_apptrace.instrumentor import setup_okahu_telemetry
from okahu_apptrace.wrap_common import llm_wrapper
from okahu_apptrace.wrapper import WrapperMethod
from helpers import OurLLM
logger = logging.getLogger(__name__)

class TestHandler(unittest.TestCase):
    @patch.object(requests.Session, 'post')
    def test_llama_index(self, mock_post):
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        os.environ["OKAHU_API_KEY"] = "key1"
        os.environ["OKAHU_INGESTION_ENDPOINT"] = "https://localhost:3000/api/v1/traces"

        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = 'mock response'

        setup_okahu_telemetry(
            workflow_name="llama_index_1",
            span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
            wrapper_methods=[
                        WrapperMethod(
                            package="helpers",
                            object="OurLLM",
                            method="complete",
                            span_name="llamaindex.OurLLM",
                            wrapper=llm_wrapper),
                        WrapperMethod(
                            package="llama_index.llms.openai.base",
                            object="OpenAI",
                            method="chat",
                            span_name="llamaindex.openai",
                            wrapper=llm_wrapper),
                    ]
            )

        llm = OurLLM()


        # check if storage already exists
        PERSIST_DIR = "./storage"
        if not os.path.exists(PERSIST_DIR):
            # load the documents and create the index
            dir_path = os.path.dirname(os.path.realpath(__file__))
            documents = SimpleDirectoryReader(dir_path + "/data").load_data()
            index = VectorStoreIndex.from_documents(documents)
            # store it for later
            index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            # load the existing index
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)


        # Either way we can now query the index
        query_engine = index.as_query_engine(llm= llm)
        query = "What did the author do growing up?"
        response = query_engine.query(query)
        time.sleep(5)
        mock_post.assert_called_once_with(
            url = 'https://localhost:3000/api/v1/traces',
            data=ANY,
            timeout=ANY
        )
        '''mock_post.call_args gives the parameters used to make post call. 
           This can be used to do more asserts'''
        dataBodyStr = mock_post.call_args.kwargs['data']
        logger.debug(dataBodyStr)
        dataJson =  json.loads(dataBodyStr) # more asserts can be added on individual fields

        root_attributes = [x for x in  dataJson["batch"] if x["parent_id"] == "None"][0]["attributes"]
        assert root_attributes["workflow_input"] == query
        assert root_attributes["workflow_output"] == llm.dummy_response

        assert len(dataJson['batch']) == 5
        # llmspan = dataJson["batch"].find

        # assert dataJson["batch"][1]["attributes"]["workflow_type"] == "workflow.llamaindex"
        span_names: List[str] = [span["name"] for span in dataJson['batch']]
        for name in ["llamaindex.retrieve", "llamaindex.query", "llamaindex.OurLLM"]:
            assert name in span_names
        
        type_found = False
        model_name_found = False

        for span in dataJson["batch"]:
            if span["name"] == "llamaindex.query" and "workflow_type" in span["attributes"]:
                assert span["attributes"]["workflow_type"] == "workflow.llamaindex"
                type_found = True
            if span["name"] == "llamaindex.OurLLM" and "openai_model_name" in span["attributes"]:
                assert span["attributes"]["openai_model_name"] == "custom"
                model_name_found = True

        assert type_found == True
        assert model_name_found == True



if __name__ == '__main__':
    unittest.main()
    
        