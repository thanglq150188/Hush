# Deploy Production

Hướng dẫn này sẽ giúp bạn deploy Hush workflows lên môi trường production một cách an toàn và hiệu quả.

## Environment Configuration

### resources.yaml cho từng environment

```bash
project/
├── configs/
│   ├── resources.dev.yaml
│   ├── resources.staging.yaml
│   └── resources.prod.yaml
```

**Development** (`resources.dev.yaml`):
```yaml
llm:default:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-mini  # Cheaper model for dev
  base_url: https://api.openai.com/v1

embedding:default:
  _class: EmbeddingConfig
  api_type: openai
  api_key: ${OPENAI_API_KEY}
  model: text-embedding-3-small
```

**Production** (`resources.prod.yaml`):
```yaml
llm:default:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o
  base_url: https://api.openai.com/v1

llm:fallback:
  _class: AzureConfig
  api_key: ${AZURE_API_KEY}
  azure_endpoint: ${AZURE_ENDPOINT}
  model: gpt-4

embedding:default:
  _class: EmbeddingConfig
  api_type: openai
  api_key: ${OPENAI_API_KEY}
  model: text-embedding-3-large
```

### Load config theo environment

```python
import os
from hush.core.registry import ResourceHub

# Set environment
env = os.getenv("HUSH_ENV", "dev")
config_path = f"configs/resources.{env}.yaml"

# Or use HUSH_CONFIG env var
os.environ["HUSH_CONFIG"] = config_path

# ResourceHub sẽ tự động load từ HUSH_CONFIG
hub = ResourceHub.instance()
```

### Environment Variables

```bash
# .env.production
HUSH_CONFIG=/app/configs/resources.prod.yaml
HUSH_TRACES_DB=/var/log/hush/traces.db

# API Keys
OPENAI_API_KEY=sk-prod-xxx
AZURE_API_KEY=xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_PUBLIC_KEY=pk-lf-xxx

# Feature flags
HUSH_TRACING=true
HUSH_LOG_LEVEL=INFO
```

### Secrets Management

**Không** commit secrets vào git. Sử dụng:

1. **Environment variables** (basic)
2. **AWS Secrets Manager / GCP Secret Manager** (recommended)
3. **HashiCorp Vault** (enterprise)

```python
# Ví dụ với AWS Secrets Manager
import boto3
import json

def get_secrets():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='hush/production')
    return json.loads(response['SecretString'])

secrets = get_secrets()
os.environ["OPENAI_API_KEY"] = secrets["openai_api_key"]
```

## Containerization (Docker)

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set environment
ENV HUSH_CONFIG=/app/configs/resources.prod.yaml
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  hush-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HUSH_CONFIG=/app/configs/resources.prod.yaml
      - HUSH_TRACING=true
    env_file:
      - .env.production
    volumes:
      - ./logs:/var/log/hush
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

## API Server Example

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from hush.core import Hush, GraphNode, START, END
from hush.providers import PromptNode, LLMNode
from hush.observability import LangfuseTracer
import os

app = FastAPI()

# Create workflow once at startup
def create_chat_workflow():
    with GraphNode(name="chat") as graph:
        prompt = PromptNode(
            name="prompt",
            inputs={
                "prompt": {"system": "You are a helpful assistant.", "user": "{query}"},
                "query": PARENT["query"]
            }
        )
        llm = LLMNode(
            name="llm",
            resource_key="llm:default",
            fallback=["llm:fallback"],
            inputs={"messages": prompt["messages"]}
        )
        llm["content"] >> PARENT["response"]
        START >> prompt >> llm >> END
    return Hush(graph)

# Initialize at startup
engine = create_chat_workflow()
tracer = LangfuseTracer(resource_key="langfuse:default") if os.getenv("HUSH_TRACING") else None

class ChatRequest(BaseModel):
    query: str
    user_id: str = None
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    request_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = await engine.run(
            inputs={"query": request.query},
            user_id=request.user_id,
            session_id=request.session_id,
            tracer=tracer
        )
        return ChatResponse(
            response=result["response"],
            request_id=result["$state"].request_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

## Monitoring

### Langfuse Integration

```python
from hush.observability import LangfuseTracer

# Production tracer
tracer = LangfuseTracer(
    resource_key="langfuse:production",
    tags=["production", "v1.2.0"]
)

result = await engine.run(
    inputs={...},
    tracer=tracer,
    user_id=user_id,
    session_id=session_id
)

# Langfuse dashboard shows:
# - Latency per node
# - Token usage
# - Cost tracking
# - Error rates
```

### OpenTelemetry Integration

```python
from hush.observability import OTelTracer

tracer = OTelTracer(resource_key="otel:jaeger")
result = await engine.run(inputs={...}, tracer=tracer)
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response

# Metrics
REQUEST_COUNT = Counter(
    'hush_requests_total',
    'Total requests',
    ['workflow', 'status']
)
REQUEST_LATENCY = Histogram(
    'hush_request_duration_seconds',
    'Request latency',
    ['workflow']
)

@app.post("/chat")
async def chat(request: ChatRequest):
    with REQUEST_LATENCY.labels(workflow="chat").time():
        try:
            result = await engine.run(...)
            REQUEST_COUNT.labels(workflow="chat", status="success").inc()
            return result
        except Exception:
            REQUEST_COUNT.labels(workflow="chat", status="error").inc()
            raise

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## Logging Best Practices

### Structured Logging

```python
import structlog
import logging

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

@app.post("/chat")
async def chat(request: ChatRequest):
    logger.info(
        "chat_request",
        user_id=request.user_id,
        query_length=len(request.query)
    )
    try:
        result = await engine.run(...)
        logger.info(
            "chat_success",
            user_id=request.user_id,
            request_id=result["$state"].request_id,
            tokens=result.get("tokens_used", {})
        )
        return result
    except Exception as e:
        logger.error(
            "chat_error",
            user_id=request.user_id,
            error=str(e),
            exc_info=True
        )
        raise
```

### Log Levels

```python
import os

LOG_LEVEL = os.getenv("HUSH_LOG_LEVEL", "INFO")

# Production: INFO hoặc WARNING
# Development: DEBUG
# Debugging issues: DEBUG với specific modules

logging.getLogger("hush").setLevel(LOG_LEVEL)
```

## Performance Tuning

### Connection Pooling

```python
import httpx

# Configure HTTP client với connection pooling
http_client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100,
        keepalive_expiry=30.0
    ),
    timeout=httpx.Timeout(30.0, connect=5.0)
)
```

### Caching

```python
from functools import lru_cache
import redis.asyncio as redis

# Redis for distributed cache
redis_client = redis.from_url(os.getenv("REDIS_URL"))

async def cached_embedding(text: str):
    cache_key = f"embedding:{hash(text)}"
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    embedding = await compute_embedding(text)
    await redis_client.setex(cache_key, 3600, json.dumps(embedding))
    return embedding
```

### Batch Processing

```python
# Batch requests để reduce API calls
from hush.core.nodes.iteration import MapNode

with MapNode(
    name="batch_process",
    inputs={"item": Each(PARENT["items"])},
    max_concurrency=10  # Control concurrency
) as map_node:
    # Process in parallel with limits
    ...
```

## Security Considerations

### Input Validation

```python
from pydantic import BaseModel, validator

class ChatRequest(BaseModel):
    query: str
    user_id: str

    @validator('query')
    def validate_query(cls, v):
        if len(v) > 10000:
            raise ValueError("Query too long")
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: ChatRequest):
    ...
```

### API Key Rotation

```yaml
# resources.yaml - support multiple keys
llm:default:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY_PRIMARY:${OPENAI_API_KEY}}
  ...
```

## Deployment Checklist

- [ ] Environment variables configured
- [ ] Secrets in secure storage (not in code)
- [ ] Health check endpoint working
- [ ] Logging configured with appropriate level
- [ ] Metrics endpoint exposed
- [ ] Tracing enabled (Langfuse/OTEL)
- [ ] Rate limiting configured
- [ ] Input validation in place
- [ ] Fallback models configured
- [ ] Error handling tested
- [ ] Connection pooling configured
- [ ] Docker image optimized
- [ ] Resource limits set

## Tiếp theo

- [Xử lý lỗi](error-handling.md) - Production error handling
- [Tracing](../concepts/tracing.md) - Detailed tracing setup
- [ResourceHub](../concepts/resource-hub.md) - Resource configuration
