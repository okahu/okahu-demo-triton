# Copyright (C) Okahu Inc 2023-2024. All rights reserved

import logging
from typing import Collection,List
from wrapt import wrap_function_wrapper
from opentelemetry.trace import get_tracer
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry import trace
from okahu_apptrace.wrapper import METHODS_LIST, WrapperMethod
from okahu_apptrace.exporter import OkahuSpanExporter 

logger = logging.getLogger(__name__)

_instruments = ("langchain >= 0.0.346",)

class OkahuInstrumentor(BaseInstrumentor):
    
    workflow_name: str = ""
    user_wrapper_methods: list[WrapperMethod] = []
    
    def __init__(
            self,
            user_wrapper_methods: list[WrapperMethod] = []) -> None:
        self.user_wrapper_methods = user_wrapper_methods
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(instrumenting_module_name= __name__, tracer_provider= tracer_provider)

        new_list = [
            {
                "package": method.package,
                "object": method.object,
                "method": method.method,
                "span_name": method.span_name,
                "wrapper": method.wrapper,
            } for method in self.user_wrapper_methods]

        final_method_list = new_list + METHODS_LIST

        for wrapped_method in final_method_list:
            try:
                wrap_package = wrapped_method.get("package")
                wrap_object = wrapped_method.get("object")
                wrap_method = wrapped_method.get("method")
                wrapper = wrapped_method.get("wrapper")
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}" if wrap_object else wrap_method,
                    wrapper(tracer, wrapped_method),
                )
            except Exception as ex:
                logger.error(f"""_instrument wrap Exception: {str(ex)} 
                             for package: {wrap_package},
                             object:{wrap_object},
                             method:{wrap_method}""")
            

    def _uninstrument(self, **kwargs):
        for wrapped_method in METHODS_LIST:
            try:
                wrap_package = wrapped_method.get("package")
                wrap_object = wrapped_method.get("object")
                wrap_method = wrapped_method.get("method")
                unwrap(
                    f"{wrap_package}.{wrap_object}" if wrap_object else wrap_package,
                    wrap_method,
                )
            except Exception as ex:
                logger.error(f"""_instrument unwrap Exception: {str(ex)} 
                             for package: {wrap_package},
                             object:{wrap_object},
                             method:{wrap_method}""")
           

def setup_okahu_telemetry(
        workflow_name: str,
        span_processors: List[SpanProcessor] = [],
        wrapper_methods: List[WrapperMethod] = []):
    resource = Resource(attributes={
        SERVICE_NAME: workflow_name
    })
    traceProvider = TracerProvider(resource=resource)
    okahuProcessor = BatchSpanProcessor(OkahuSpanExporter())
    for processor in span_processors:
        traceProvider.add_span_processor(processor)
    traceProvider.add_span_processor(okahuProcessor)
    trace.set_tracer_provider(traceProvider)
    instrumentor = OkahuInstrumentor(user_wrapper_methods=wrapper_methods)
    instrumentor.app_name = workflow_name
    instrumentor.instrument()


