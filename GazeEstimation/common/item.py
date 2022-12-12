from pydantic import BaseModel

class Item(BaseModel):
    meetingId: str
    name: str
    data: str
    startTimestamp: int
    endTimestamp: int
    values: list = []