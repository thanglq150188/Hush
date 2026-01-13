"""Opik configuration for ResourceHub."""

from typing import Optional

from hush.core.utils.yaml_model import YamlModel


class OpikConfig(YamlModel):
    """Configuration for Opik observability backend.

    Opik is an open-source LLM observability platform by Comet.
    This config is registered to ResourceHub and used to create OpikClient.

    Attributes:
        api_key: API key for Opik authentication (required for Comet cloud)
        workspace: Workspace name for organizing projects
        project_name: Default project name for traces
        host: Opik server URL (for self-hosted instances)
        enabled: Whether tracing is enabled
        sample_rate: Sampling rate for traces (0.0 to 1.0)

    Example:
        ```yaml
        # resources.yaml - Comet cloud
        opik:default:
          _class: OpikConfig
          api_key: ${OPIK_API_KEY}
          workspace: my-workspace
          project_name: my-project

        # resources.yaml - Self-hosted
        opik:local:
          _class: OpikConfig
          host: http://localhost:5173
          project_name: my-project
        ```

        ```python
        from hush.core.registry import get_hub

        client = get_hub().opik("default")
        ```
    """

    api_key: Optional[str] = None
    workspace: Optional[str] = None
    project_name: Optional[str] = None
    host: Optional[str] = None
    enabled: bool = True
    sample_rate: float = 1.0

    @classmethod
    def from_env(cls) -> "OpikConfig":
        """Create config from environment variables.

        Environment variables:
            - OPIK_API_KEY (optional for self-hosted)
            - OPIK_WORKSPACE (optional)
            - OPIK_PROJECT_NAME (optional)
            - OPIK_HOST (optional, for self-hosted)
        """
        import os

        return cls(
            api_key=os.environ.get("OPIK_API_KEY"),
            workspace=os.environ.get("OPIK_WORKSPACE"),
            project_name=os.environ.get("OPIK_PROJECT_NAME"),
            host=os.environ.get("OPIK_HOST"),
        )

    @classmethod
    def local(cls, project_name: str = "default") -> "OpikConfig":
        """Create config for local/self-hosted Opik instance.

        Args:
            project_name: Project name for traces

        Returns:
            OpikConfig configured for local instance
        """
        return cls(
            host="http://localhost:5173",
            project_name=project_name,
        )
