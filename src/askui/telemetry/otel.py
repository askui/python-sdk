from fastapi import FastAPI
from opentelemetry import trace
from pydantic import BaseModel, Field, SecretStr, model_validator
from typing_extensions import Self

from askui import __version__


class OtelSettings(BaseModel):
    """Settings for otel configuration"""

    enabled: bool = Field(default=False)
    secret: SecretStr | None = Field(
        default=None,
        description="Secret for OTLP authentication. Required when enabled=True.",
    )
    service_name: str = Field(default="chat-api")
    service_version: str = Field(default=__version__)
    endpoint: str | None = Field(
        default=None,
        description="OTLP endpoint URL.",
    )
    cluster_name: str = Field(default="askui-dev")

    @model_validator(mode="after")
    def validate_secret_when_enabled(self) -> Self:
        """Ensure secret is provided when OpenTelemetry is enabled."""
        if self.enabled and self.secret is None:
            error_msg = "Secret is required when OpenTelemetry is enabled"
            raise ValueError(error_msg)
        return self


def setup_opentelemetry_tracing(app: FastAPI, settings: OtelSettings) -> None:
    """
    Set up OpenTelemetry tracing for the FastAPI application.

    Args:
        app (FastAPI): The FastAPI application to instrument for tracing.
        settings (OtelSettings): OpenTelemetry configuration settings containing
            endpoint, secret, service name, and version.

    Returns:
        None

    """
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import (  # type: ignore[import-not-found]
            FastAPIInstrumentor,
        )
        from opentelemetry.instrumentation.httpx import (  # type: ignore[import-not-found]
            HTTPXClientInstrumentor,
        )
        from opentelemetry.instrumentation.sqlalchemy import (  # type: ignore[import-not-found]
            SQLAlchemyInstrumentor,
        )
        from opentelemetry.sdk.resources import (  # type: ignore[import-not-found]
            Resource,
        )
        from opentelemetry.sdk.trace import (  # type: ignore[import-not-found]
            TracerProvider,
        )
        from opentelemetry.sdk.trace.export import (  # type: ignore[import-not-found]
            BatchSpanProcessor,
        )
    except ImportError:
        return

    resource = Resource.create(
        {
            "service.name": settings.service_name,
            "service.version": settings.service_version,
            "cluster.name": settings.cluster_name,
        }
    )
    provider = TracerProvider(resource=resource)

    otlp_exporter = OTLPSpanExporter(
        endpoint=settings.endpoint,
        headers={"authorization": f"Basic {settings.secret.get_secret_value()}"},  # type: ignore[union-attr]
    )

    span_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(span_processor)

    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app, excluded_urls="health")
    HTTPXClientInstrumentor().instrument()
    SQLAlchemyInstrumentor().instrument()
