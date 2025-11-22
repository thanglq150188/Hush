"""Base configuration for tracers."""

from hush.core.utils.yaml_model import YamlModel


class TracerConfig(YamlModel):
    """Base configuration class for all tracer backends.

    All tracer config classes (LangfuseConfig, PhoenixConfig, etc.)
    should inherit from this base class to enable proper plugin registration.
    """
    pass
