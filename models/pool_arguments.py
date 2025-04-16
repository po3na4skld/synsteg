from pydantic import BaseModel


class PoolArguments(BaseModel):
    container_split: str | list[str]
    bits_per_word: int


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
        binary_message_chunks: list[str]
    ):
        return cls(
            pool_arguments=[PoolArguments(
                container_split=container_split,
                bits_per_word=bits_per_word
            ) for container_split in container_splits],
            additional_bits=additional_bits,
            binary_message_chunks=binary_message_chunks,
        )
