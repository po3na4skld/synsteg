from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Which model writes containers and synonym tables. model=None keeps utils.constants (gpt-4o)."""
    provider: str = "openai"  # "openai" or "ollama"
    model: str | None = None
    url: str = "http://localhost:11434"  # ollama only
    api_key: str | None = Field(None, repr=False)  # OpenAI; None = OPENAI_API_KEY from the environment


class PoolArguments(BaseModel):
    container_split: str | list[str]
    bits_per_word: int
    llm: LLMConfig = LLMConfig()


class SecretKeyGenerationBody(BaseModel):
    pool_arguments: list[PoolArguments]
    additional_bits: int
    binary_message_chunks: list[str] = []

    @classmethod
    def from_list(
        cls,
        container_splits: list[str],
        bits_per_word: int,
        additional_bits: int,
        binary_message_chunks: list[str],
        llm: LLMConfig = LLMConfig(),
    ):
        return cls(
            pool_arguments=[PoolArguments(
                container_split=container_split,
                bits_per_word=bits_per_word,
                llm=llm,
            ) for container_split in container_splits],
            additional_bits=additional_bits,
            binary_message_chunks=binary_message_chunks,
        )
