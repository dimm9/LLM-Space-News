from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, List


class CalcArgs(BaseModel):
    a: float
    b: float


class RagArgs(BaseModel):
    query: str = Field(..., max_length=300, description="Question to the knowledge base")
    top_k: int = Field(3, ge=1, le=10, description="Number of fragments to retrieve")


class ConvertArgs(BaseModel):
    value: float
    from_unit: Literal["km", "mi", "c", "f"]
    to_unit: Literal["km", "mi", "c", "f"]


class SearchArgs(BaseModel):
    pattern: str = Field(..., max_length=64)


class ToolCall(BaseModel):
    tool: Literal[
        "calculator.add", "calculator.sub", "calculator.mul", "calculator.div", "units.convert", "files.search", "kb.lookup"]
    args: Dict[str, Any]


ALLOWED_TOOLS = {
    "calculator.add", "calculator.sub", "calculator.mul", "calculator.div",
    "units.convert", "files.search", "kb.lookup"
}
