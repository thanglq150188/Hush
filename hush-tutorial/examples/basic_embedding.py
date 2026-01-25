"""Basic embedding example using Hush.

This example demonstrates:
1. Setting up ResourceHub from a YAML config
2. Creating a simple embedding workflow
3. Running embeddings on text inputs
"""

import asyncio
from hush.core import WorkflowEngine, START, END, INPUT, OUTPUT, set_global_hub
from hush.core.registry import ResourceHub
from hush.providers import EmbeddingNode  # Plugin auto-registers!


async def main():
    # 1. Setup ResourceHub from YAML configuration
    # This assumes you have a resources.yaml file in the parent directory
    # Plugins are already auto-registered when EmbeddingNode is imported!
    hub = ResourceHub.from_yaml("../resources.yaml")

    # 2. Set the global ResourceHub instance
    set_global_hub(hub)

    # 4. Create a simple embedding workflow
    with WorkflowEngine(name="simple_embed") as workflow:
        embed = EmbeddingNode(
            name="embed_texts",
            resource_key="bge-m3",  # References the embedding model in resources.yaml
            inputs={"texts": INPUT},
            outputs={"embeddings": OUTPUT}
        )

        START >> embed >> END

    # 5. Compile the workflow
    workflow.compile()

    # 6. Run the workflow with sample texts
    sample_texts = [
        "Hello world",
        "How are you today?",
        "This is a test of the embedding system"
    ]

    print("Running embedding workflow...")
    print(f"Input texts: {sample_texts}")

    result = await workflow.run(inputs={"texts": sample_texts})

    # 7. Display results
    embeddings = result["embeddings"]
    print(f"\nGenerated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
    print(f"First embedding (truncated): {embeddings[0][:5]}..." if embeddings else "No embeddings")

    return result


if __name__ == "__main__":
    asyncio.run(main())
