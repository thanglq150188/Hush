"""Phoenix configuration for ResourceHub.

Arize Phoenix is an open-source AI observability platform that provides
tracing, evaluation, and debugging capabilities for LLM applications.

Phoenix uses OpenTelemetry under the hood and can run locally or in the cloud.

References:
    - Documentation: https://arize.com/docs/phoenix
    - GitHub: https://github.com/Arize-ai/phoenix
"""

from typing import Dict, Optional

from hush.core.utils.yaml_model import YamlModel


class PhoenixConfig(YamlModel):
    """Configuration for Phoenix observability backend.

    Phoenix can be configured for:
    - Local development (default localhost:6006)
    - Phoenix Cloud (with API key)
    - Self-hosted deployment

    Example in resources.yaml:
        ```yaml
        # Local Phoenix
        phoenix:local:
          _class: PhoenixConfig
          endpoint: http://localhost:6006/v1/traces
          project_name: my-project

        # Phoenix Cloud
        phoenix:cloud:
          _class: PhoenixConfig
          endpoint: https://app.phoenix.arize.com/v1/traces
          api_key: your-api-key
          project_name: my-project
        ```

    Attributes:
        endpoint: The collector endpoint for traces.
            - Local gRPC: http://localhost:4317 (default)
            - Local HTTP: http://localhost:6006/v1/traces
            - Cloud: https://app.phoenix.arize.com/v1/traces
        project_name: Name of the project to associate traces with.
        api_key: API key for Phoenix Cloud authentication (optional for local).
        headers: Additional headers to include in requests.
        batch: Whether to batch spans before sending (recommended for production).
        enabled: Whether tracing is enabled.
        sample_rate: Fraction of traces to sample (0.0 to 1.0).
    """

    endpoint: str = "http://localhost:4317"
    project_name: str = "default"
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    batch: bool = True
    enabled: bool = True
    sample_rate: float = 1.0

    def get_headers(self) -> Dict[str, str]:
        """Get headers including API key if configured.

        Returns:
            Dictionary of headers for Phoenix requests.
        """
        result = dict(self.headers) if self.headers else {}
        if self.api_key:
            result["Authorization"] = f"Bearer {self.api_key}"
        return result
