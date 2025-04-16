import json

from models.base import BaseModel


class ReportModel(BaseModel):
    fields = [
        "uuid",
        "created",
        "modified",
        "message",
        "secret_key",
        "container",
        "encoded_message",
        "encoding_time_report",
        "encoding_usage_report",
        "decoding_time",
        "decoded_message",
        "error_message"
    ]

    collection_name = "reports"

    def __init__(self, **kwargs):
        super(ReportModel, self).__init__(**kwargs)
        self.message = kwargs["message"]
        self.container = kwargs["container"]
        self.encoded_message = kwargs["encoded_message"]
        self.secret_key = kwargs["secret_key"]
        self.encoding_time_report = kwargs["encoding_time_report"]
        self.encoding_usage_report = kwargs["encoding_usage_report"]
        self.decoding_time = kwargs.get("decoding_time", "")
        self.decoded_message = kwargs.get("decoded_message", "")
        self.error_message = kwargs.get("error_message", "")

    def to_json(self, base_path: str):
        with open(f"{base_path}/{self.uuid}.json", "w") as f:
            json.dump({k: getattr(self, k) for k in self.fields if k not in ["created", "modified"]}, f)
