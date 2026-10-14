from pydantic import BaseModel, HttpUrl


class RequestBody(BaseModel):
    url: HttpUrl


class SuccessResponse(BaseModel):
    message: str
    requestId: str
    url: str
    status: str


class ErrorResponse(BaseModel):
    error: str
