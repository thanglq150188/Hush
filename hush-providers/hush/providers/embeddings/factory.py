from hush.providers.embeddings.base import BaseEmbedder
from hush.providers.embeddings.tei import TEIEmbedding
from hush.providers.embeddings.vllm import VLLMEmbedding
from hush.providers.embeddings.huggingface import HFEmbedding
from hush.providers.embeddings.onnx import ONNXEmbedding
from .config import (
    EmbeddingConfig,
    EmbeddingType
)



class EmbeddingFactory:
    r"""Factory class for embedding model backends

    Raises:
        ValueError: in case the provided model type is unknown.
    """

    @staticmethod
    def create(
        config: EmbeddingConfig
    ) -> BaseEmbedder:
        if config.api_type == EmbeddingType.TEXT_EMBEDDING_INFERENCE:
            model_class = TEIEmbedding
        elif config.api_type == EmbeddingType.VLLM:
            model_class = VLLMEmbedding
        elif config.api_type == EmbeddingType.HF:
            model_class = HFEmbedding
        elif config.api_type == EmbeddingType.ONNX:
            model_class = ONNXEmbedding
        else:
            raise ValueError(f"Unsupported Model: {config.api_type}")
        return model_class(config)


async def main():

    embed = EmbeddingFactory.create(
        config=EmbeddingConfig.default()
    )

    # Test with Vietnamese text
    test_text = "tổng giám đốc của MB là ai ?"
    vectors = await embed.run(test_text)
    print(f"Generated embedding vectors: {vectors}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
