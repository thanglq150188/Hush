"""Tests for hush-providers nodes using ResourceHub."""

import pytest
from pathlib import Path

from hush.core.registry import ResourceHub
from hush.core import WorkflowEngine, START, END, INPUT, OUTPUT
from hush.providers import (
    LLMNode,
    EmbeddingNode,
    RerankNode,
    LLMPlugin,
    EmbeddingPlugin,
    RerankPlugin,
)
from hush.providers.llms.config import OpenAIConfig
from hush.providers.embeddings.config import EmbeddingConfig
from hush.providers.rerankers.config import RerankingConfig


@pytest.fixture
def hub():
    """Create ResourceHub with test configurations."""
    hub = ResourceHub.from_memory()

    # Register plugins
    hub.register_plugins(LLMPlugin, EmbeddingPlugin, RerankPlugin)

    # Add test LLM config
    llm_config = OpenAIConfig(
        api_type="openai",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        model="gpt-4"
    )
    hub.register(llm_config, registry_key="llm:gpt-4", persist=False)

    # Add test embedding config
    embed_config = EmbeddingConfig(
        api_type="hf",
        model="BAAI/bge-m3",
        dimensions=1024
    )
    hub.register(embed_config, registry_key="embedding:bge-m3", persist=False)

    # Add test reranking config
    rerank_config = RerankingConfig(
        api_type="hf",
        model="BAAI/bge-reranker-v2-m3"
    )
    hub.register(rerank_config, registry_key="reranking:bge-m3", persist=False)

    # Set as singleton
    ResourceHub.set_instance(hub)

    return hub


def test_hub_has_resources(hub):
    """Test that hub has registered resources."""
    assert hub.has("llm:gpt-4")
    assert hub.has("embedding:bge-m3")
    assert hub.has("reranking:bge-m3")


def test_llm_node_initialization(hub):
    """Test LLMNode can be initialized with ResourceHub."""
    node = LLMNode(
        name="test_llm",
        resource_key="gpt-4",
        inputs={"messages": INPUT},
        outputs={"content": OUTPUT}
    )

    assert node.name == "test_llm"
    assert node.resource_key == "gpt-4"
    assert node.type == "llm"
    assert hasattr(node, 'core')


def test_llm_node_in_workflow(hub):
    """Test LLMNode can be added to workflow."""
    with WorkflowEngine(name="test_workflow") as workflow:
        llm = LLMNode(
            name="chat",
            resource_key="gpt-4",
            inputs={"messages": INPUT},
            outputs={"content": OUTPUT}
        )
        START >> llm >> END

    workflow.compile()

    # Check workflow structure
    assert len(workflow._graph.nodes) > 0
    assert "chat" in [node.name for node in workflow._graph.nodes.values()]


def test_embedding_node_initialization(hub):
    """Test EmbeddingNode can be initialized with ResourceHub."""
    node = EmbeddingNode(
        name="test_embed",
        resource_key="bge-m3",
        inputs={"texts": INPUT},
        outputs={"embeddings": OUTPUT}
    )

    assert node.name == "test_embed"
    assert node.resource_key == "bge-m3"
    assert node.type == "embedding"
    assert hasattr(node, 'core')


def test_embedding_node_in_workflow(hub):
    """Test EmbeddingNode can be added to workflow."""
    with WorkflowEngine(name="embed_workflow") as workflow:
        embed = EmbeddingNode(
            name="embed",
            resource_key="bge-m3",
            inputs={"texts": INPUT},
            outputs={"embeddings": OUTPUT}
        )
        START >> embed >> END

    workflow.compile()

    assert len(workflow._graph.nodes) > 0
    assert "embed" in [node.name for node in workflow._graph.nodes.values()]


def test_rerank_node_initialization(hub):
    """Test RerankNode can be initialized with ResourceHub."""
    node = RerankNode(
        name="test_rerank",
        resource_key="bge-m3",
        inputs={"query": INPUT, "documents": INPUT},
        outputs={"reranks": OUTPUT}
    )

    assert node.name == "test_rerank"
    assert node.resource_key == "bge-m3"
    assert node.type == "rerank"
    assert hasattr(node, 'core')
    assert hasattr(node, 'backend')


def test_rerank_node_in_workflow(hub):
    """Test RerankNode can be added to workflow."""
    with WorkflowEngine(name="rerank_workflow") as workflow:
        rerank = RerankNode(
            name="rerank",
            resource_key="bge-m3",
            inputs={"query": INPUT, "documents": INPUT, "top_k": INPUT},
            outputs={"reranks": OUTPUT}
        )
        START >> rerank >> END

    workflow.compile()

    assert len(workflow._graph.nodes) > 0
    assert "rerank" in [node.name for node in workflow._graph.nodes.values()]


def test_multi_node_workflow(hub):
    """Test workflow with multiple node types."""
    with WorkflowEngine(name="multi_node") as workflow:
        embed = EmbeddingNode(
            name="embed",
            resource_key="bge-m3",
            inputs={"texts": INPUT},
            outputs={"embeddings": "vectors"}
        )

        rerank = RerankNode(
            name="rerank",
            resource_key="bge-m3",
            inputs={"query": INPUT, "documents": INPUT, "top_k": 3},
            outputs={"reranks": "ranked_docs"}
        )

        llm = LLMNode(
            name="generate",
            resource_key="gpt-4",
            inputs={"messages": INPUT},
            outputs={"content": OUTPUT}
        )

        START >> embed >> rerank >> llm >> END

    workflow.compile()

    # Check all nodes are in workflow
    node_names = [node.name for node in workflow._graph.nodes.values()]
    assert "embed" in node_names
    assert "rerank" in node_names
    assert "generate" in node_names


def test_node_metadata(hub):
    """Test nodes return correct metadata."""
    llm = LLMNode(name="llm", resource_key="gpt-4", inputs=INPUT, outputs=OUTPUT)
    embed = EmbeddingNode(name="embed", resource_key="bge-m3", inputs=INPUT, outputs=OUTPUT)
    rerank = RerankNode(name="rerank", resource_key="bge-m3", inputs=INPUT, outputs=OUTPUT)

    assert llm.specific_metadata() == {"model": "gpt-4"}
    assert embed.specific_metadata() == {"model": "bge-m3"}
    assert rerank.specific_metadata() == {"model": "bge-m3"}


def test_llm_node_streaming_mode(hub):
    """Test LLMNode in streaming mode."""
    node = LLMNode(
        name="streaming_llm",
        resource_key="gpt-4",
        inputs={"messages": INPUT},
        outputs={"content": OUTPUT},
        stream=True
    )

    assert node.stream is True
    assert node.resource_key == "gpt-4"


def test_llm_node_instant_response(hub):
    """Test LLMNode with instant response."""
    node = LLMNode(
        name="instant_llm",
        resource_key="gpt-4",
        inputs={"messages": INPUT},
        outputs={"content": OUTPUT},
        instant_response=True
    )

    assert node.instant_response is True
    # instant_response nodes don't set self.core
    assert not hasattr(node, 'core') or node.core is None


def test_resource_hub_cleanup(hub):
    """Test ResourceHub can be cleaned up."""
    # Clear singleton
    ResourceHub._instance = None

    # Should raise error if accessing singleton
    with pytest.raises(RuntimeError):
        ResourceHub.instance()


def test_nodes_with_file_config():
    """Test nodes work with file-based ResourceHub."""
    # Create a temporary config file
    import tempfile
    import yaml

    config_data = {
        "llm:test-model": {
            "_class": "OpenAIConfig",
            "api_type": "openai",
            "api_key": "test-key",
            "model": "gpt-4",
            "base_url": "https://api.openai.com/v1"
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_file = f.name

    try:
        # Create hub from file
        hub = ResourceHub.from_yaml(config_file)
        hub.register_plugin(LLMPlugin)
        ResourceHub.set_instance(hub)

        # Create node
        node = LLMNode(
            name="test",
            resource_key="test-model",
            inputs=INPUT,
            outputs=OUTPUT
        )

        assert node.resource_key == "test-model"

    finally:
        # Cleanup
        Path(config_file).unlink()
        ResourceHub._instance = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
