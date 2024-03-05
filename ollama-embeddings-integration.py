from typing import Type, List

from cat.mad_hatter.decorators import tool, hook, plugin
from cat.factory.embedder import EmbedderSettings
from langchain_community.embeddings import OllamaEmbeddings
from pydantic import ConfigDict


class OllamaEmbedderConfig(EmbedderSettings):
    base_url: str = "http://ollama:11434"
    model: str='nomic-embed-text'
    _pyclass: Type = OllamaEmbeddings

    model_config = ConfigDict(
        json_schema_extra = {
            "humanReadableName": "Ollama Embedder",
            "description": "A high-performing open embedding model with 8192 tokens context window",
            "link": "https://ollama.com/library/nomic-embed-text",
        }
    )


@hook
def factory_allowed_embedders(allowed, cat) -> List:
    allowed.append(OllamaEmbedderConfig)
    return allowed
