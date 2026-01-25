# Hush Engine

<!--
MỤC ĐÍCH: API reference cho Hush class
NỘI DUNG SẼ VIẾT:
- Class signature: Hush(graph: GraphNode)
- Properties:
  - name: str
  - schema: StateSchema
- Methods:
  - run(inputs, tracer, user_id, session_id, request_id) -> Dict
  - __call__(inputs) -> Dict (alias cho run)
  - show() -> None
- Return value structure: {"result": ..., "$state": MemoryState}
- Code examples
-->
