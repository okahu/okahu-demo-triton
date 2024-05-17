#!/bin/bash

if [[ $# -lt 1 ]]; then
    echo "Usage <Okahu_key>"
    exit 1
fi

echo Configuring chatbot apps with local Triton inference server
sed 's/OKAHU_API_KEY=/OKAHU_API_KEY='"$1"'/g ; s/TRITON_LLM_ENDPOINT=/TRITON_LLM_ENDPOINT=http:\/\/127.0.0.1:8000\/v2\/models\/flan_t5\/infer/g ; s/TRITON_LLM_GRPC_ENDPOINT=/TRITON_LLM_GRPC_ENDPOINT=http:\/\/127.0.0.1:8001' config/config.ini.template > config/config.ini 
TRITON_LLM_GRPC_ENDPOINT

