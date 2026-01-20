from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

from app.router import NasaAgent
from app.llm_engine import chat_once

app = FastAPI(title="NASA AI Agent", description="An AI agent built as part of an LLM project, designed to answer questions and perform reasoning tasks using NASA-related data.")

print("Server Start: Loading Agent...")
agent = NasaAgent()

class AskRequest(BaseModel):
    query: str
    use_functions: bool = True
    k: int = 3
    mode: str = "standard"
    model_mode: Literal["gemini", "groq", "local"] = "gemini"

class AskResponse(BaseModel):
    answer: str
    tool_used: Optional[str] = None
    status: str

def mode_to_max_tokens(mode: str) -> int:
    mode = (mode or "standard").lower()
    return {"concise": 220, "standard": 520, "detailed": 1000}.get(mode, 520)

def mode_to_style_hint(mode: str) -> str:
    mode = (mode or "standard").lower()
    if mode == "concise":
        return "Be concise. Use short bullet points. Avoid extra detail."
    if mode == "detailed":
        return "Be detailed and structured. Explain clearly and include context."
    return "Answer normally with a balanced level of detail."


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest):
    if not req.use_functions:
        try:
            style = mode_to_style_hint(req.mode)
            resp = chat_once(
                req.query,
                system=f"You are a helpful assistant. {style}",
                temperature=0.7,
                max_output_tokens=mode_to_max_tokens(req.mode),
                model_mode=req.model_mode
            )
            return {
                "answer": resp['text'],
                "tool_used": None,
                "status": "ok_chat_only"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    try:
        result = agent.run(req.query, model_mode=req.model_mode, mode=req.mode, top_k=req.k)
        status = result.get("status", "ok")
        answer_txt = (result.get("answer") or "")
        if "blocked" in status or answer_txt.startswith("security_blocked:"):
            raise HTTPException(
                status_code=403,
                detail=f"Safety Block: {answer_txt}"
            )
        if status == "error":
            low = answer_txt.lower()
            if "quota" in low or "resource_exhausted" in low or "429" in low:
                raise HTTPException(
                    status_code=429,
                    detail="LLM rate limit exceeded (Gemini quota). Please retry later or switch model."
                )
            if "timeout" in low or "timed out" in low:
                raise HTTPException(
                    status_code=504,
                    detail=f"Timeout: {answer_txt}"
                )
            if answer_txt.startswith("validation:"):
                raise HTTPException(
                    status_code=422,
                    detail=f"Validation error: {answer_txt}"
                )
            if answer_txt.startswith("security_blocked:"):
                raise HTTPException(
                    status_code=403,
                    detail=f"Safety Block: {answer_txt}"
                )
            raise HTTPException(
                status_code=500,
                detail=f"Tool Error: {answer_txt}"
            )
        return {
            "answer": answer_txt,
            "tool_used": result.get("tool_used"),
            "status": status
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# to check if works
@app.get("/health")
def health():
    return {"status": "running"}

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def ui_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)