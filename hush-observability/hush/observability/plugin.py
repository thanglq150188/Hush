"""Plugin for auto-registering observability backends to ResourceHub.

This plugin registers config classes and factory handlers for all
observability backends, allowing them to be loaded from resources.yaml.

Example:
    ```yaml
    # resources.yaml
    langfuse:vpbank:
      _class: LangfuseConfig
      public_key: pk-...
      secret_key: sk-...
      host: https://cloud.langfuse.com
    ```

    ```python
    # Auto-registration happens on import
    import hush.observability

    from hush.core.registry import get_hub
    client = get_hub().langfuse("vpbank")
    ```
"""

from hush.core.loggings import LOGGER


class ObservabilityPlugin:
    """Plugin for registering observability backends to ResourceHub."""

    _registered = False

    @classmethod
    def register(cls):
        """Register all observability config classes and factory handlers."""
        if cls._registered:
            return

        try:
            from hush.core.registry import (
                register_config_class,
                register_factory_handler,
            )

            # Register Langfuse
            from hush.observability.backends.langfuse import (
                LangfuseConfig,
                LangfuseClient,
            )

            register_config_class(LangfuseConfig)
            register_factory_handler(LangfuseConfig, lambda c: LangfuseClient(c))

            # Register Opik
            from hush.observability.backends.opik import (
                OpikConfig,
                OpikClient,
            )

            register_config_class(OpikConfig)
            register_factory_handler(OpikConfig, lambda c: OpikClient(c))

            # Register OpenTelemetry
            from hush.observability.backends.otel import (
                OTELConfig,
                OTELClient,
            )

            register_config_class(OTELConfig)
            register_factory_handler(OTELConfig, lambda c: OTELClient(c))

            # Register Phoenix
            from hush.observability.backends.phoenix import (
                PhoenixConfig,
                PhoenixClient,
            )

            register_config_class(PhoenixConfig)
            register_factory_handler(PhoenixConfig, lambda c: PhoenixClient(c))

            cls._registered = True
            LOGGER.debug("ObservabilityPlugin registered successfully")

        except ImportError as e:
            LOGGER.warning(
                "Failed to register ObservabilityPlugin: %s. "
                "ResourceHub integration will not be available.",
                str(e),
            )

    @classmethod
    def is_registered(cls) -> bool:
        """Check if plugin is registered."""
        return cls._registered


# Auto-register on import
ObservabilityPlugin.register()
