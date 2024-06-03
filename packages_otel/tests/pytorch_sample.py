# Copyright (C) Okahu Inc 2023-2024. All rights reserved

import os
import torch
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

from okahu_apptrace.instrumentor import setup_okahu_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from okahu_apptrace.wrapper import WrapperMethod,llm_wrapper


import json
import os.path
import time
import unittest
from unittest.mock import ANY, patch
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
import requests

class TestHandler(unittest.TestCase):
    @patch.object(requests.Session, 'post')
    def test_pytorch(self, mock_post):
        os.environ["OKAHU_API_KEY"] = "key1"
        os.environ["OKAHU_INGESTION_ENDPOINT"] = "https://localhost:3000/api/v1/traces"
        os.environ["OPENAI_API_KEY"] = ""

        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = 'mock response'

        setup_okahu_telemetry(
            workflow_name="pytorch_1",
            span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
            wrapper_methods=[
                        WrapperMethod(
                            package="transformers",
                            object="GPT2DoubleHeadsModel",
                            method="forward",
                            span_name="pytorch.transformer.GPT2DoubleHeadsModel",
                            wrapper=llm_wrapper),
                    ]
            )

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

        # Add a [CLS] to the vocabulary (we should train it also!)
        num_added_tokens = tokenizer.add_special_tokens({'cls_token': '[CLS]'})

        embedding_layer = model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size

        choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        encoded_choices = [tokenizer.encode(s) for s in choices]
        cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

        input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1
        # the trace gets generated for the forward method which gets called here
        outputs = model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

        time.sleep(5)
        mock_post.assert_called_once_with(
            url = 'https://localhost:3000/api/v1/traces',
            data=ANY,
            timeout=ANY
        )
        '''mock_post.call_args gives the parameters used to make post call. 
           This can be used to do more asserts'''
        dataBodyStr = mock_post.call_args.kwargs['data']
        dataJson =  json.loads(dataBodyStr) # more asserts can be added on individual fields
        assert len(dataJson['batch']) == 1
        assert dataJson['batch'][0]["name"] == "pytorch.transformer.GPT2DoubleHeadsModel"

if __name__ == '__main__':
    unittest.main()
        