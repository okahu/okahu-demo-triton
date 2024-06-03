# Copyright (C) Okahu Inc 2023-2024. All rights reserved


import json
import logging
import os
from typing import Optional, Sequence
import requests
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan

REQUESTS_SUCCESS_STATUS_CODES = (200, 202)

logger = logging.getLogger(__name__)

class OkahuSpanExporter(SpanExporter):
    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout: Optional[int] = None,
        session: Optional[requests.Session] = None,
    ):
        """Okahu exporter."""
        okahu_endpoint: str = os.environ["OKAHU_INGESTION_ENDPOINT"]
        self.endpoint = endpoint or okahu_endpoint
        
        api_key: str = os.environ["OKAHU_API_KEY"]

        self.session = session or requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "x-api-key": api_key}
        )
        self._closed = False
        self.timeout = timeout or 10

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        # After the call to Shutdown subsequent calls to Export are
        # not allowed and should return a Failure result
        if self._closed:
            logger.warning("Exporter already shutdown, ignoring batch")
            return SpanExportResult.FAILUREencoder
        
        if len(spans) == 0:
            return

        span_list = {
            "batch" : []
        }

        # append the batch object with all the spans object
        for span in spans:
            # create a object from serialized span
            obj = json.loads(span.to_json())
            if obj["parent_id"] is None:
                obj["parent_id"] = "None"
            else:
                obj["parent_id"]=obj["parent_id"].strip('0x')
            if obj["context"] is not None:
                obj["context"]["trace_id"] = obj["context"]["trace_id"].strip('0x')
                obj["context"]["span_id"] = obj["context"]["span_id"].strip('0x')
            span_list["batch"].append(obj)

        result = self.session.post(
            url=self.endpoint,
            data=json.dumps(span_list),
            timeout=self.timeout,
        )
        if result.status_code not in REQUESTS_SUCCESS_STATUS_CODES:
            logger.error(
                "Traces cannot be uploaded; status code: %s, message %s",
                result.status_code,
                result.text,
            )
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        if self._closed:
            logger.warning("Exporter already shutdown, ignoring call")
            return
        self.session.close()
        self._closed = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True