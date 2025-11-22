"""Advanced workflow example using multiple Hush providers.

This example demonstrates:
1. Combining LLM, embedding, and reranking nodes in a single workflow
2. Complex data flow between nodes
3. Using different provider types together
4. Real-world RAG (Retrieval Augmented Generation) pattern
"""

import asyncio
from hush.core import WorkflowEngine, START, END, INPUT, OUTPUT, BaseNode
from hush.core.registry import ResourceHub
from hush.providers import LLMNode, EmbeddingNode, RerankNode
from hush.providers import LLMPlugin, EmbeddingPlugin, RerankPlugin


async def main():
    # 1. Setup ResourceHub with all plugins
    hub = ResourceHub.from_yaml("../resources.yaml")

    # Register all provider plugins
    hub.register_plugin(LLMPlugin)
    hub.register_plugin(EmbeddingPlugin)
    hub.register_plugin(RerankPlugin)

    ResourceHub.set_instance(hub)

    # 2. Create an advanced RAG workflow
    # This workflow:
    # - Embeds the user query
    # - Embeds document chunks
    # - Reranks documents based on relevance
    # - Generates an answer using the LLM with top documents

    with WorkflowEngine(name="rag_workflow") as workflow:
        # Node 1: Embed the user query
        embed_query = EmbeddingNode(
            name="embed_query",
            resource_key="bge-m3",
            inputs={"texts": INPUT["query"]},
            outputs={"embeddings": "query_embedding"}
        )

        # Node 2: Embed document chunks
        embed_docs = EmbeddingNode(
            name="embed_docs",
            resource_key="bge-m3",
            inputs={"texts": INPUT["documents"]},
            outputs={"embeddings": "doc_embeddings"}
        )

        # Node 3: Rerank documents based on query
        rerank = RerankNode(
            name="rerank_docs",
            resource_key="bge-m3",
            inputs={
                "query": INPUT["query"],
                "documents": INPUT["documents"],
                "top_k": 3
            },
            outputs={"reranks": "ranked_docs"}
        )

        # Node 4: Format context for LLM
        # This is a simple custom node that formats the reranked documents
        class FormatContextNode(BaseNode):
            async def core(self, ranked_docs, query):
                # Format top documents into context
                context = "\n\n".join([
                    f"Document {i+1} (score: {doc['score']:.3f}):\n{doc['content']}"
                    for i, doc in enumerate(ranked_docs)
                ])

                # Create messages for LLM
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer the user's question based on the provided context."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}"
                    }
                ]

                return {"messages": messages}

        format_context = FormatContextNode(
            name="format_context",
            inputs={
                "ranked_docs": "ranked_docs",
                "query": INPUT["query"]
            },
            outputs={"messages": "formatted_messages"}
        )

        # Node 5: Generate answer using LLM
        llm = LLMNode(
            name="generate_answer",
            resource_key="gpt-4",
            inputs={"messages": "formatted_messages"},
            outputs={"content": OUTPUT["answer"]},
            stream=False
        )

        # Define workflow flow
        START >> embed_query
        START >> embed_docs
        START >> rerank >> format_context >> llm >> END

    # 3. Compile the workflow
    workflow.compile()

    # 4. Prepare sample data
    sample_documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
        "Neural networks are computational models inspired by biological neural networks.",
        "Hush is a workflow orchestration framework for building complex async pipelines.",
        "Docker is a platform for developing, shipping, and running applications in containers."
    ]

    sample_query = "What is Hush and how does it help with workflows?"

    # 5. Run the workflow
    print("Running advanced RAG workflow...")
    print(f"Query: {sample_query}")
    print(f"Document count: {len(sample_documents)}")

    result = await workflow.run(inputs={
        "query": sample_query,
        "documents": sample_documents
    })

    # 6. Display results
    print("\n" + "="*80)
    print("WORKFLOW RESULTS")
    print("="*80)

    if "answer" in result:
        print(f"\nGenerated Answer:\n{result['answer']}")

    if "ranked_docs" in result:
        print("\n" + "-"*80)
        print("Top Ranked Documents:")
        for i, doc in enumerate(result["ranked_docs"][:3]):
            print(f"\n{i+1}. Score: {doc['score']:.4f}")
            print(f"   Content: {doc['content'][:100]}...")

    return result


async def run_simple_llm_example():
    """Simple LLM example for testing."""
    print("\n" + "="*80)
    print("SIMPLE LLM EXAMPLE")
    print("="*80)

    hub = ResourceHub.from_yaml("../resources.yaml")
    hub.register_plugin(LLMPlugin)
    ResourceHub.set_instance(hub)

    with WorkflowEngine(name="simple_chat") as workflow:
        llm = LLMNode(
            name="chat",
            resource_key="gpt-4",
            inputs={"messages": INPUT},
            outputs={"content": OUTPUT},
            stream=False
        )

        START >> llm >> END

    workflow.compile()

    messages = [
        {"role": "user", "content": "Explain what a workflow engine is in one sentence."}
    ]

    result = await workflow.run(inputs={"messages": messages})
    print(f"\nResponse: {result.get('content', 'No response')}")

    return result


if __name__ == "__main__":
    # Run the advanced workflow
    asyncio.run(main())

    # Uncomment to run simple LLM example
    # asyncio.run(run_simple_llm_example())
