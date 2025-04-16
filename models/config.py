import json

from models.base import BaseModel


class Config(BaseModel):

    fields = [
        "uuid",
        "created",
        "modified",
        "openai_api_key",
        "bits_per_word",
        "additional_bits",
        "mongodb",
    ]

    collection_name = "configs"

    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.openai_api_key = kwargs["openai_api_key"]
        self.bits_per_word = kwargs["bits_per_word"]
        self.additional_bits = kwargs["additional_bits"]

        self.mongodb = kwargs["mongodb"]

    @classmethod
    def from_json(cls, path_to_json: str):
        with open(path_to_json) as cfg:
            return cls(**json.load(cfg))
