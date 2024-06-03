# Copyright (C) Okahu Inc 2023-2024. All rights reserved

from okahu_apptrace.wrap_common import atask_wrapper, task_wrapper, llm_wrapper


LLAMAINDEX_METHODS = [
    {
        "package": "llama_index.core.indices.base_retriever",
        "object": "BaseRetriever",
        "method": "retrieve",
        "span_name": "llamaindex.retrieve",
        "wrapper": task_wrapper
    },
    {
        "package": "llama_index.core.indices.base_retriever",
        "object": "BaseRetriever",
        "method": "aretrieve",
        "span_name": "llamaindex.retrieve",
        "wrapper": atask_wrapper
    },
    {
        "package": "llama_index.core.base.base_query_engine",
        "object": "BaseQueryEngine",
        "method": "query",
        "span_name": "llamaindex.query",
        "wrapper": task_wrapper,
    },
    {
        "package": "llama_index.core.base.base_query_engine",
        "object": "BaseQueryEngine",
        "method": "aquery",
        "span_name": "llamaindex.query",
        "wrapper": atask_wrapper,
    },
    {
        "package": "llama_index.core.llms.custom",
        "object": "CustomLLM",
        "method": "chat",
        "span_name": "llamaindex.llmchat",
        "wrapper": task_wrapper,
    },
    {
        "package": "llama_index.core.llms.custom",
        "object": "CustomLLM",
        "method": "achat",
        "span_name": "llamaindex.llmchat",
        "wrapper": atask_wrapper,
    },
    {
        "package": "triton_custom_llm",
        "object": "TritonLlamaLLM",
        "method": "complete",
        "span_name": "tritonllm",
        "wrapper": llm_wrapper,
    }
]