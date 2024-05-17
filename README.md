# okahu-demo
This repo includes demo chat applications.

## Prerequisits
- OpenAI subscription and API key for OpenAI
Please refer to openai.com for details
- Okahu observability cloud subscription and API key
Visit okahu.ai to signup and get API key
- NVIDIA Triton Inference server.
  - To setup your own cloud/self hosted setup, please refer to triton-setup\README.md for details. This is required for Coffee chatbot application that uses inference model hosted on Triton.
  - Docker setup on your local machine (Linux, Mac, Windows). Please refer to setup instructions on docker.com

## Configuring demo environment
- Go to folder config
- Copy config.ini.template to config.ini
- Edit the file and add OpenAI API Key and Okahu API key
- Set the Triton inference server endpoint if you have a Triton inference server configured.


## Chatbot client using NVIDIA Triton inference server
This application uses RAG design pattern to facilitates a coffee chat bot. It's a python program that uses Langchain library. The vector dataset is built using multi-qa-mpnet-base-dot-v1 from Huggingface from a set of Wikipedia articles. The vector data is stored in a local filebased faiss vectorDB. The app uses flan_t5 model for inference that's hosted on a Triton inference server instance.
### Coffee chatbot app with Okahu instrumentation
To run the command line coffee chatbot app use following command from the top level directory
```./coffee_chatbot.sh```

## Using Okahu's demo docker with NVIDIA Triton inference server setup
### Download and run Okahu demo container
- Download the container
  ```docker pull okahudocker/okahu_demo:okahu_triton_llama_index_demo```
- Start container
  ```docker run --rm -p8000:8000 -p8001:8001 -p8002:8002  okahudocker/okahu_demo:okahu_triton_llama_index_demo <Okahu-API-Key> <OpenAI-API-Key> ```
- Verify the container running
  ``` docker ps ```  
  Note the container ID retured by above command where the Image name is okahudocker/okahu_triton_apps_demo
### Coffee chatbot app with Okahu instrumentation
To run the command line coffee chatbot app with Okahu langchain log handler, use following command from the top level
``` docker exec -it <Container-ID> bash /okahu_demo/coffee_chatbot.sh ``` 
