# LLM Providers

<!--
MỤC ĐÍCH: API reference cho LLM classes
NỘI DUNG SẼ VIẾT:
- BaseLLM (abstract):
  - chat(messages) -> Response
  - generate(prompt) -> str
- OpenAISDKModel:
  - Constructor: OpenAISDKModel(config: OpenAIConfig)
  - Supported models
- OpenAIConfig:
  - Fields: api_key, base_url, model, api_type, proxy, cost_per_token
- AzureOpenAI specific config
- GeminiModel và GeminiConfig
- Response structure
-->
