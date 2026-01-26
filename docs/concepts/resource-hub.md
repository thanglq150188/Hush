# ResourceHub

## Vấn đề

Trong các ứng dụng AI/LLM thực tế, bạn cần quản lý nhiều loại resources:
- Nhiều LLM providers (OpenAI, Azure, Gemini, local models)
- Database connections (Redis, MongoDB, Milvus)
- External services (S3, Langfuse, MCP servers)

Hardcode credentials và configs vào code tạo ra nhiều vấn đề:
- Secrets bị commit vào git
- Khó switch giữa dev/staging/production
- Duplicate config ở nhiều nơi

## ResourceHub là gì?

**ResourceHub** là centralized registry để quản lý tất cả resources của ứng dụng:

- **Centralized config**: Tất cả configs trong 1 file YAML
- **Lazy loading**: Resources chỉ được khởi tạo khi truy cập lần đầu
- **Environment variable interpolation**: `${VAR}` hoặc `${VAR:default}`
- **Typed accessors**: `hub.llm()`, `hub.embedding()`, `hub.reranker()`
- **Plugin system**: Packages bên ngoài có thể đăng ký resource types mới
- **Health check**: Kiểm tra trạng thái các resources

## Cấu hình với resources.yaml

### Tạo file resources.yaml

```yaml
# LLM configurations
llm:gpt-4o:
  type: openai
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o
  base_url: https://api.openai.com/v1

llm:azure-gpt-4:
  type: azure
  api_key: ${AZURE_API_KEY}
  endpoint: ${AZURE_ENDPOINT}
  deployment_name: gpt-4
  api_version: "2024-02-15-preview"

# Embedding configuration
embedding:bge-m3:
  type: embedding
  api_key: ${EMBEDDING_API_KEY}
  api_type: vllm
  base_url: https://api.deepinfra.com/v1/openai/embeddings
  model: BAAI/bge-m3
  dimensions: 1024

# Observability
langfuse:default:
  type: LangfuseConfig
  secret_key: ${LANGFUSE_SECRET_KEY}
  public_key: ${LANGFUSE_PUBLIC_KEY}
  host: https://cloud.langfuse.com
  enabled: true
```

### Các field bắt buộc

- `type`: Loại resource (bắt buộc) - Hush dùng để biết cách parse và khởi tạo resource. Có thể là:
  - Type alias: `openai`, `azure`, `gemini`, `embedding`, `reranking`
  - Class name: `OpenAIConfig`, `AzureConfig`, `LangfuseConfig`

### Environment Variable Interpolation

ResourceHub hỗ trợ 2 cú pháp:

| Syntax | Mô tả | Ví dụ |
|--------|-------|-------|
| `${VAR}` | Required - warning nếu không set | `api_key: ${OPENAI_API_KEY}` |
| `${VAR:default}` | Optional - dùng default nếu không set | `host: ${REDIS_HOST:localhost}` |

## Đặt biến môi trường HUSH_CONFIG

Hush tìm config theo thứ tự:

1. Biến môi trường `HUSH_CONFIG`
2. `./resources.yaml` (thư mục hiện tại)
3. `~/.hush/resources.yaml` (thư mục home)

```bash
# Linux/macOS
export HUSH_CONFIG=/path/to/your/resources.yaml

# Windows
set HUSH_CONFIG=C:\path\to\your\resources.yaml
```

## Sử dụng ResourceHub

### Global Hub

```python
from hush.core.registry import get_hub

hub = get_hub()

# Typed accessors
llm = hub.llm("gpt-4o")              # BaseLLM instance
embedding = hub.embedding("bge-m3")  # BaseEmbedding instance
```

### Custom Hub

```python
from hush.core.registry import ResourceHub, set_global_hub

# Tạo hub từ file config cụ thể
hub = ResourceHub.from_yaml("/path/to/config.yaml")

# Thay thế global hub
set_global_hub(hub)
```

## Typed Accessors

```python
# LLM - tự động thêm prefix "llm:"
llm = hub.llm("gpt-4o")           # → hub.get("llm:gpt-4o")

# Embedding - tự động thêm prefix "embedding:"
embedding = hub.embedding("bge-m3")  # → hub.get("embedding:bge-m3")

# Reranker - tự động thêm prefix "reranking:"
reranker = hub.reranker("bge-reranker")  # → hub.get("reranking:bge-reranker")

# Các services khác
redis = hub.redis("default")
langfuse = hub.langfuse("default")
```

## Health Check

```python
result = hub.health_check()

print(result.healthy)   # True nếu tất cả OK
print(result.passed)    # List keys đã pass
print(result.failed)    # List keys đã fail
print(result.errors)    # Dict {key: error_message}
```

## Đăng ký Resource Types Mới

ResourceHub sử dụng hệ thống plugin để mở rộng. Mỗi config class cần khai báo `_type` và `_category`, sau đó đăng ký với `REGISTRY`.

### Cách đăng ký

```python
from typing import ClassVar
from hush.core.registry import REGISTRY
from hush.core.utils.yaml_model import YamlModel


# 1. Định nghĩa Config class với _type và _category
class MyServiceConfig(YamlModel):
    """Config cho MyService."""
    _type: ClassVar[str] = "my-service"        # Type alias trong YAML
    _category: ClassVar[str] = "myservice"     # Category/namespace

    api_key: str
    endpoint: str
    timeout: int = 30


# 2. Định nghĩa factory function
def create_my_service(config: MyServiceConfig):
    """Factory function tạo MyService instance."""
    return MyServiceClient(
        api_key=config.api_key,
        endpoint=config.endpoint,
        timeout=config.timeout
    )


# 3. Đăng ký với REGISTRY (chỉ 1 call)
REGISTRY.register(MyServiceConfig, create_my_service)
```

Sau đó có thể sử dụng trong `resources.yaml`:

```yaml
myservice:production:
  type: my-service
  api_key: ${MY_SERVICE_API_KEY}
  endpoint: https://api.myservice.com
  timeout: 60
```

```python
# Sử dụng
from hush.core.registry import get_hub
service = get_hub().get("myservice:production")
```

### Plugin Pattern (Recommended)

Phù hợp cho packages có nhiều config types:

```python
# my_package/registry/plugin.py

from hush.core.registry import REGISTRY
from my_package.configs import MyServiceConfig, MyServiceV2Config
from my_package.factory import MyServiceFactory


class MyServicePlugin:
    """Plugin đăng ký MyService resources với ResourceHub.

    Gọi MyServicePlugin.register() một lần khi startup.
    """

    _registered = False

    @classmethod
    def register(cls):
        """Đăng ký tất cả config classes với factory."""
        if cls._registered:
            return

        # Chỉ cần 1 call cho mỗi config class
        REGISTRY.register(MyServiceConfig, MyServiceFactory.create)
        REGISTRY.register(MyServiceV2Config, MyServiceFactory.create_v2)

        cls._registered = True

    @classmethod
    def is_registered(cls) -> bool:
        """Kiểm tra plugin đã được đăng ký chưa."""
        return cls._registered


# Auto-register khi import module
MyServicePlugin.register()
```

**Ví dụ thực tế từ hush-providers:**

```python
# hush-providers/hush/providers/registry/llm_plugin.py

from hush.core.registry import REGISTRY
from hush.providers.llms.config import LLMConfig, OpenAIConfig, AzureConfig, GeminiConfig
from hush.providers.llms.factory import LLMFactory


class LLMPlugin:
    """Plugin đăng ký LLM resources với ResourceHub."""

    _registered = False

    @classmethod
    def register(cls):
        if cls._registered:
            return

        # Đăng ký tất cả config classes với factory
        REGISTRY.register(LLMConfig, LLMFactory.create)
        REGISTRY.register(OpenAIConfig, LLMFactory.create)
        REGISTRY.register(AzureConfig, LLMFactory.create)
        REGISTRY.register(GeminiConfig, LLMFactory.create)

        cls._registered = True


# Auto-register on import
LLMPlugin.register()
```

**Ví dụ từ hush-observability:**

```python
# hush-observability/hush/observability/plugin.py

class ObservabilityPlugin:
    """Plugin đăng ký observability backends với ResourceHub."""

    _registered = False

    @classmethod
    def register(cls):
        if cls._registered:
            return

        try:
            from hush.core.registry import REGISTRY

            # Register Langfuse
            from hush.observability.backends.langfuse import LangfuseConfig, LangfuseClient
            REGISTRY.register(LangfuseConfig, lambda c: LangfuseClient(c))

            # Register OpenTelemetry
            from hush.observability.backends.otel import OTELConfig, OTELClient
            REGISTRY.register(OTELConfig, lambda c: OTELClient(c))

            cls._registered = True

        except ImportError:
            # Graceful degradation nếu dependencies không có
            pass


# Auto-register on import
ObservabilityPlugin.register()
```

### Cấu trúc file cho Plugin

```
my_package/
├── __init__.py
├── configs/
│   ├── __init__.py
│   └── my_config.py      # Config classes với _type, _category
├── factory.py            # Factory functions
└── registry/
    ├── __init__.py       # Export plugins
    └── my_plugin.py      # Plugin class với REGISTRY.register()
```

## Sử dụng trong Workflow

### Với LLMNode

```python
from hush.providers.nodes import LLMNode, PromptNode

llm = LLMNode(
    name="llm",
    resource="llm:gpt-4o",  # Tham chiếu key trong resources.yaml
    inputs={
        "messages": prompt["messages"],
        "temperature": 0.7
    }
)
```

## Production Best Practices

### 1. Tách config theo môi trường

```bash
# Development
export HUSH_CONFIG=./configs/resources.dev.yaml

# Production
export HUSH_CONFIG=./configs/resources.prod.yaml
```

### 2. Sử dụng .env file

```python
from dotenv import load_dotenv
load_dotenv()

from hush.core.registry import get_hub
hub = get_hub()
```

### 3. Validation trước khi chạy

```python
result = hub.health_check(keys=["llm:gpt-4o", "embedding:bge-m3"])
if not result.healthy:
    raise RuntimeError(f"Missing resources: {result.failed}")
```

### 4. Không commit secrets

```yaml
# .gitignore
resources.yaml
*.env
```

## Tiếp theo

- [Tracing](tracing.md) - Debug và monitor workflows
