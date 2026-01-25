# Kiến trúc Internal

<!--
MỤC ĐÍCH: Giải thích internal architecture cho contributors
NỘI DUNG SẼ VIẾT:
- Package structure:
  - hush-core: engine, nodes, state, registry, tracers
  - hush-providers: LLM, embedding, reranking implementations
  - hush-observability: external tracing adapters
- Key classes và responsibilities
- Execution flow: Hush.run() internals
- ResourceFactory pattern
- Adding new node types
- Adding new providers
- Testing strategy
-->
