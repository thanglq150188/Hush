# Plan: Restructure Documentation

> Big update - Tái cấu trúc toàn bộ documentation system

## Vấn đề hiện tại

### 1. Không phân biệt User docs vs Internal docs
- `docs/concepts/` mix giữa cách dùng và kiến trúc internal
- `docs/architecture/` trùng lặp với `docs/concepts/`
- User phải đọc implementation details để hiểu cách dùng

### 2. README.md duplicate với docs
- Main README.md trống
- Subproject READMEs duplicate nội dung với docs
- Gây confusion không biết đọc ở đâu

### 3. User không biết bắt đầu từ đâu
- `docs/index.md` chỉ là skeleton
- Không có learning path rõ ràng
- Quá nhiều entry points

---

## Cấu trúc mới

```
hush/
├── README.md                         ← Minimal: overview + pip install + link docs
│
├── docs/                             ← USER DOCS (cách SỬ DỤNG Hush)
│   ├── index.md                      ← Landing page với learning path
│   ├── installation.md
│   ├── quickstart.md
│   │
│   ├── tutorials/                    ← Step-by-step, theo thứ tự
│   │   ├── 01-first-workflow.md
│   │   ├── 02-llm-basics.md
│   │   ├── 03-loops-branches.md
│   │   └── 04-production.md
│   │
│   ├── guides/                       ← Task-oriented, đọc độc lập
│   │   ├── llm-integration.md
│   │   ├── embeddings-reranking.md
│   │   ├── error-handling.md
│   │   ├── parallel-execution.md
│   │   └── tracing.md
│   │
│   ├── examples/                     ← Complete, copy-paste được
│   │   ├── rag-workflow.md
│   │   ├── agent-workflow.md
│   │   └── multi-model.md
│   │
│   └── api/                          ← API reference (optional, có thể auto-gen)
│       ├── core.md
│       ├── providers.md
│       └── observability.md
│
├── architecture/                     ← INTERNAL DOCS (cho dev/AI hiểu engine)
│   ├── index.md                      ← Overview + reading order
│   │
│   ├── engine/                       ← Core execution engine
│   │   ├── execution-flow.md         ← Workflow chạy như thế nào
│   │   ├── compilation.md            ← Graph compilation process
│   │   └── scheduling.md             ← Node scheduling & dependency resolution
│   │
│   ├── state/                        ← State management system
│   │   ├── overview.md               ← State system overview
│   │   ├── state-schema.md           ← StateSchema design
│   │   ├── memory-state.md           ← MemoryState implementation
│   │   ├── indexer.md                ← WorkflowIndexer internals
│   │   └── data-flow.md              ← Cách data flow qua nodes
│   │
│   ├── nodes/                        ← Node system
│   │   ├── base-node.md              ← BaseNode anatomy
│   │   ├── graph-node.md             ← Nested graphs & scoping
│   │   ├── iteration-nodes.md        ← ForLoop, Map, While internals
│   │   ├── branch-node.md            ← Conditional routing
│   │   └── creating-custom-node.md   ← Guide tạo node mới
│   │
│   ├── resources/                    ← Resource management
│   │   ├── resource-hub.md           ← ResourceHub design & singleton
│   │   ├── plugin-system.md          ← Plugin architecture
│   │   └── config-loading.md         ← YAML parsing & env interpolation
│   │
│   ├── tracing/                      ← Observability internals
│   │   ├── tracer-interface.md       ← BaseTracer abstract design
│   │   ├── local-tracer.md           ← SQLite implementation details
│   │   ├── trace-data-model.md       ← Database schema & relationships
│   │   └── async-buffer.md           ← AsyncTraceBuffer design
│   │
│   ├── providers/                    ← Provider system
│   │   ├── llm-abstraction.md        ← LLM provider interface
│   │   ├── embedding-provider.md     ← Embedding provider design
│   │   ├── reranker-provider.md      ← Reranker design
│   │   └── adding-new-provider.md    ← Guide thêm provider mới
│   │
│   └── contributing/                 ← Contribution guides
│       ├── development-setup.md      ← Setup dev environment
│       ├── code-style.md             ← Coding conventions
│       ├── testing.md                ← Testing strategy
│       └── release-process.md        ← Release workflow
│
└── hush-*/
    └── README.md                     ← Minimal: pip install + 1 example
```

---

## Nguyên tắc phân biệt

### docs/ (User Documentation)

| Aspect | Description |
|--------|-------------|
| **Audience** | End users, application developers |
| **Purpose** | Học cách SỬ DỤNG Hush |
| **Tone** | Friendly, task-oriented |
| **Content** | What to do, not how it works internally |
| **Examples** | "Gọi LLM như thế nào" |

### architecture/ (Internal Documentation)

| Aspect | Description |
|--------|-------------|
| **Audience** | Core developers, AI assistants (Claude, Cursor) |
| **Purpose** | Hiểu cách Hush HOẠT ĐỘNG bên trong |
| **Tone** | Technical, implementation-focused |
| **Content** | Design decisions, data structures, algorithms |
| **Examples** | "LLMNode.execute() hoạt động như thế nào" |

---

## Migration Plan

### Phase 1: Tạo cấu trúc mới

```bash
# Tạo folders
mkdir -p docs/tutorials docs/guides docs/examples docs/api
mkdir -p architecture/engine architecture/state architecture/nodes
mkdir -p architecture/resources architecture/tracing architecture/providers
mkdir -p architecture/contributing
```

### Phase 2: Di chuyển files từ docs/ cũ

| File cũ | File mới | Action |
|---------|----------|--------|
| `docs/index.md` | `docs/index.md` | Rewrite hoàn toàn |
| `docs/getting-started/installation.md` | `docs/installation.md` | Move + simplify |
| `docs/getting-started/quickstart.md` | `docs/quickstart.md` | Move |
| `docs/getting-started/first-workflow.md` | `docs/tutorials/01-first-workflow.md` | Move + rename |
| `docs/concepts/overview.md` | `architecture/index.md` | Move (internal content) |
| `docs/concepts/graph-and-nodes.md` | `architecture/nodes/base-node.md` | Split |
| `docs/concepts/state-management.md` | `architecture/state/overview.md` | Move |
| `docs/concepts/tracing.md` | Split | User part → `docs/guides/tracing.md`, Internal → `architecture/tracing/` |
| `docs/concepts/resource-hub.md` | `architecture/resources/resource-hub.md` | Move |
| `docs/guides/building-workflows.md` | `docs/tutorials/` | Split into tutorials |
| `docs/guides/llm-integration.md` | `docs/guides/llm-integration.md` | Keep, simplify |
| `docs/guides/embeddings-reranking.md` | `docs/guides/embeddings-reranking.md` | Keep |
| `docs/guides/error-handling.md` | `docs/guides/error-handling.md` | Keep, simplify |
| `docs/guides/parallel-execution.md` | `docs/guides/parallel-execution.md` | Keep |
| `docs/guides/production-deployment.md` | `docs/tutorials/04-production.md` | Move |
| `docs/examples/*` | `docs/examples/*` | Keep |
| `docs/architecture/*` | `architecture/` | Merge vào architecture/ |
| `docs/reference/*` | Delete hoặc `docs/api/` | Skeleton, xóa hoặc auto-gen |
| `docs/contributing/*` | `architecture/contributing/` | Move |
| `docs/migration-to-rust.md` | `architecture/` hoặc delete | Decide later |

### Phase 3: Xóa folders cũ

```bash
# Sau khi migrate xong
rm -rf docs/getting-started
rm -rf docs/concepts
rm -rf docs/architecture
rm -rf docs/reference
rm -rf docs/contributing
```

### Phase 4: Update README files

| File | Action |
|------|--------|
| `README.md` | Rewrite - minimal, link to docs |
| `hush-core/README.md` | Simplify - just pip install + 1 example |
| `hush-providers/README.md` | Simplify |
| `hush-observability/README.md` | Simplify |
| `hush-ai/README.md` | Simplify |
| `hush-vscode-traceview/README.md` | Keep (đã viết mới) |

### Phase 5: Viết content mới

| File | Priority | Notes |
|------|----------|-------|
| `docs/index.md` | HIGH | Landing page với learning path |
| `architecture/index.md` | HIGH | Overview cho devs/AI |
| `docs/tutorials/02-llm-basics.md` | MEDIUM | Tách từ guides |
| `docs/tutorials/03-loops-branches.md` | MEDIUM | Tách từ building-workflows |
| `architecture/engine/execution-flow.md` | HIGH | Critical cho AI hiểu |
| `architecture/state/data-flow.md` | HIGH | Critical cho AI hiểu |
| `architecture/nodes/creating-custom-node.md` | MEDIUM | Cho contributors |

---

## docs/index.md Template

```markdown
# Hush Documentation

> Async workflow orchestration engine cho GenAI applications.

## Bắt đầu từ đây 🚀

| Step | Thời gian | Link |
|------|-----------|------|
| 1. Cài đặt | 2 phút | [Installation](installation.md) |
| 2. Hello World | 5 phút | [Quickstart](quickstart.md) |
| 3. Workflow đầu tiên | 15 phút | [Tutorial](tutorials/01-first-workflow.md) |

## Tutorials (theo thứ tự)

1. [Workflow đầu tiên](tutorials/01-first-workflow.md) - Cơ bản về nodes và edges
2. [Sử dụng LLM](tutorials/02-llm-basics.md) - PromptNode và LLMNode
3. [Loops và Branches](tutorials/03-loops-branches.md) - Flow control
4. [Production](tutorials/04-production.md) - Tracing, error handling, deployment

## Guides (đọc khi cần)

- [Tích hợp LLM](guides/llm-integration.md)
- [Embeddings & Reranking](guides/embeddings-reranking.md)
- [Xử lý lỗi](guides/error-handling.md)
- [Thực thi song song](guides/parallel-execution.md)
- [Tracing & Debug](guides/tracing.md)

## Examples

- [RAG Pipeline](examples/rag-workflow.md) - Retrieval-Augmented Generation
- [AI Agent](examples/agent-workflow.md) - Agent với tools
- [Multi-model](examples/multi-model.md) - Nhiều LLM providers

## Cho Developers

Nếu bạn muốn hiểu cách Hush hoạt động bên trong hoặc contribute:
→ [Architecture Documentation](../architecture/index.md)
```

---

## architecture/index.md Template

```markdown
# Hush Architecture

> Tài liệu này dành cho core developers và AI assistants
> để hiểu cách Hush hoạt động bên trong.

## Tổng quan hệ thống

```
┌─────────────────────────────────────────────────────────┐
│                      User Code                          │
│         (GraphNode, CodeNode, LLMNode, ...)             │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    Hush Engine                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Compilation │  │  Execution  │  │  Scheduling │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   State System                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ StateSchema │  │ MemoryState │  │   Indexer   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## Reading Order (đọc theo thứ tự)

### Level 1: Core Concepts
1. [Execution Flow](engine/execution-flow.md) - Workflow chạy như thế nào
2. [State Overview](state/overview.md) - State system basics
3. [Node Anatomy](nodes/base-node.md) - Cấu trúc một node

### Level 2: Deep Dive
4. [Data Flow](state/data-flow.md) - Cách data di chuyển qua nodes
5. [Graph Compilation](engine/compilation.md) - Build process
6. [Iteration Nodes](nodes/iteration-nodes.md) - ForLoop, Map, While

### Level 3: Advanced
7. [ResourceHub](resources/resource-hub.md) - Resource management
8. [Tracer System](tracing/tracer-interface.md) - Observability

## Quick Reference

### Muốn hiểu X hoạt động như thế nào?

| Topic | File |
|-------|------|
| Workflow execution | [engine/execution-flow.md](engine/execution-flow.md) |
| State management | [state/overview.md](state/overview.md) |
| Node lifecycle | [nodes/base-node.md](nodes/base-node.md) |
| Nested graphs | [nodes/graph-node.md](nodes/graph-node.md) |
| Loops | [nodes/iteration-nodes.md](nodes/iteration-nodes.md) |
| Tracing | [tracing/tracer-interface.md](tracing/tracer-interface.md) |

### Muốn contribute/extend?

| Task | File |
|------|------|
| Tạo custom node | [nodes/creating-custom-node.md](nodes/creating-custom-node.md) |
| Thêm LLM provider | [providers/adding-new-provider.md](providers/adding-new-provider.md) |
| Setup dev environment | [contributing/development-setup.md](contributing/development-setup.md) |

## Packages

| Package | Mô tả | Key Files |
|---------|-------|-----------|
| hush-core | Workflow engine | `engine.py`, `nodes/`, `states/` |
| hush-providers | LLM/Embedding nodes | `nodes/llm.py`, `nodes/embedding.py` |
| hush-observability | Tracing backends | `tracers/`, `buffer.py` |
```

---

## Checklist

- [ ] Tạo folder structure mới
- [ ] Migrate files theo bảng
- [ ] Viết `docs/index.md` mới
- [ ] Viết `architecture/index.md` mới
- [ ] Simplify tất cả README.md
- [ ] Xóa folders cũ
- [ ] Test tất cả links
- [ ] Update any cross-references

---

## Notes

- Giữ nguyên ngôn ngữ tiếng Việt
- Code examples giữ tiếng Anh (variable names, comments)
- Priority: `docs/index.md` và `architecture/index.md` là quan trọng nhất
- Có thể làm incremental - không cần hoàn thành tất cả cùng lúc
