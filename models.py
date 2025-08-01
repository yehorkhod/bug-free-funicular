from pydantic import BaseModel


class DialogMessage(BaseModel):
    speaker: str
    content: str


class UploadTextRequest(BaseModel):
    dialog: list[DialogMessage]
    speaker: str


class TrainRequest(BaseModel):
    speaker: str
