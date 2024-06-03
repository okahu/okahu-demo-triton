# Okahu callback handler

This package provides okahu telemetry setup.

## Installing the package
```
> python3 -m pip install pipenv

> pipenv install okahu-observability
```

## References

[Managing application dependencies](https://packaging.python.org/en/latest/tutorials/managing-dependencies/)

## Usage
```python
from okahu_apptrace.instrumentor import setup_okahu_telemetry
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# Set the OKAHU_API_KEY environment variable, if not set already
os.environ["OKAHU_API_KEY"] = "okh_XXXXXXXX_XXXXXXXXXXXXXXXXXXXXXX"

# Call the setup Okahu telemetry method
app_name = "simple_math_app"
setup_okahu_telemetry(workflow_name=app_name)

llm = OpenAI()
prompt = PromptTemplate.from_template("1 + {number} = ")

chain = LLMChain(llm=llm, prompt=prompt)
chain.invoke({"number":2})

# Request callbacks: Finally, let's use the request `callbacks` to achieve the same result
chain = LLMChain(llm=llm, prompt=prompt)
chain.invoke({"number":2}, {"callbacks":[handler]})
    
```

### Monitoring custom methods with Okahu

```python
from okahu_apptrace.wrapper import WrapperMethod,task_wrapper,atask_wrapper
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# extend the default wrapped methods list as follows
app_name = "simple_math_app"
setup_okahu_telemetry(
        workflow_name=app_name,
        span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
        wrapper_methods=[
            WrapperMethod(
                package="langchain.schema.runnable",
                object="RunnableParallel",
                method="invoke",
                span_name="langchain.workflow",
                wrapper=task_wrapper),
            WrapperMethod(
                package="langchain.schema.runnable",
                object="RunnableParallel",
                method="ainvoke",
                span_name="langchain.workflow",
                wrapper=atask_wrapper)
        ])

```

