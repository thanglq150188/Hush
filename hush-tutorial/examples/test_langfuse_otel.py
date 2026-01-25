"""Test pushing OTEL traces directly to Langfuse OTEL endpoint."""

import base64
import time

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Langfuse credentials
PUBLIC_KEY = "pk-lf-2edb3eb7-c4a5-4264-b41b-01e91dc91801"
SECRET_KEY = "sk-lf-06032612-18ac-48d8-9a13-6aeaa9f42076"
HOST = "https://langfuse.aws.coreai.vpbank.dev"

# Base64 encode credentials
auth = base64.b64encode(f"{PUBLIC_KEY}:{SECRET_KEY}".encode()).decode()

# Setup OTEL exporter pointing to Langfuse
exporter = OTLPSpanExporter(
    endpoint=f"{HOST}/api/public/otel/v1/traces",
    headers={"Authorization": f"Basic {auth}"},
)

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("langfuse-otel-test")


def main():
    print(f"Sending OTEL traces to: {HOST}/api/public/otel/v1/traces")

    # Create a parent span (will appear as trace in Langfuse)
    with tracer.start_as_current_span("my-workflow") as parent:
        parent.set_attribute("user_id", "test-user-123")
        parent.set_attribute("environment", "testing")

        # Child span 1
        with tracer.start_as_current_span("step-1-fetch-data") as span1:
            span1.set_attribute("source", "database")
            time.sleep(0.1)  # Simulate work
            span1.set_attribute("records_fetched", 42)

        # Child span 2
        with tracer.start_as_current_span("step-2-process") as span2:
            span2.set_attribute("algorithm", "transform-v2")
            time.sleep(0.05)

            # Nested child
            with tracer.start_as_current_span("step-2a-validate") as span2a:
                span2a.set_attribute("validation_passed", True)
                time.sleep(0.02)

        # Child span 3 - simulate LLM call
        with tracer.start_as_current_span("step-3-llm-call") as span3:
            span3.set_attribute("gen_ai.system", "openai")
            span3.set_attribute("gen_ai.request.model", "gpt-4")
            span3.set_attribute("gen_ai.usage.prompt_tokens", 150)
            span3.set_attribute("gen_ai.usage.completion_tokens", 50)
            time.sleep(0.1)

    # Force flush to ensure traces are sent
    provider.force_flush()

    print("Done! Check Langfuse UI for traces.")


if __name__ == "__main__":
    main()
