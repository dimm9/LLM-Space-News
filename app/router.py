import json
import re
from typing import Dict, Any, Optional

from app.llm_engine import chat_once
from app.tools.registry import ToolCall
from app.tools.handlers import run_tool_safe
from app.guardrails import (
    detect_injection,
    scrub_user_input,
    contains_pii,
    contains_profanity,
    links_not_allowed, contains_path_traversal
)

SYSTEM_ROUTER = """
You are the NASA Knowledge Assistant.
Your goal is to answer user questions using available tools.

TOOLS ALLOWED:
1. "kb.lookup" - Use for ANY questions about space, NASA, rockets, science, definitions, missions, news.
   Args: {"query": "string (keywords)", "top_k": 3}
2. "calculator.add" / "sub" / "mul" / "div" - Use for math operations.
   Args: {"a": float, "b": float}
3. "units.convert" - Use for unit conversion (km/mi, c/f).
   Args: {"value": float, "from_unit": "...", "to_unit": "..."}
4. "files.search"
    Use ONLY if the user asks to search local documents/files by name/pattern.
    Required args:
    {"pattern": string}
    
INSTRUCTIONS:
- If none of the non-default rules match, use "kb.lookup".
- Never invent numbers or units if the user didn't ask about numbers or units.
- Never leave args incomplete.
- If using a tool: output ONLY JSON. If NOT using a tool: output plain text.
- If the user asks a question that requires external knowledge or calculation, output a JSON tool call.
- If the user greets you or asks something trivial, reply with plain text.
- STRICT JSON FORMAT for tools (no markdown, no comments):
  {"tool": "kb.lookup", "args": {"query": "...", "top_k": 3}}
- If the answer is not explicitly stated in the context, say: "I did not find this in the database."
- Always include at least one 'Source:' line.
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

def _fallback_kb_toolcall(user_query: str, top_k: int = 3) -> ToolCall:
    return ToolCall(tool="kb.lookup", args={"query": user_query, "top_k": top_k})

def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    for m in re.findall(r"\{.*?\}", text, re.DOTALL):
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None

def _make_repair_prompt(bad_text: str) -> str:
    return (
        "Your previous output was NOT valid JSON.\n"
        "Return ONLY a valid JSON object.\n"
        "Schema:\n"
        "{\"tool\": \"<tool_name>\", \"args\": { ... }}\n"
        "No markdown, no code fences, no comments, no extra text.\n\n"
        "Invalid output was:\n"
        f"{bad_text}"
    )
def _truncate(text: str, max_chars: int) -> str:
    text = text or ""
    return text if len(text) <= max_chars else text[:max_chars] + "\n…[truncated]"

def _validate_kb_result(result: Dict[str, Any]) -> str | None:
    if not isinstance(result, dict):
        return "validation: invalid_kb_response_not_dict"
    if result.get("error"):
        return f"validation: kb_error_returned ({result.get('error')})"
    if result.get("found") is False:
        return None
    hits = result.get("hits")
    if not isinstance(hits, list):
        return "validation: invalid_kb_response_hits_not_list"
    for i, h in enumerate(hits[:10]):
        doc = None
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            doc = h[1]
        elif isinstance(h, dict):
            doc = h
        if doc is None or not isinstance(doc, dict):
            return f"validation: invalid_kb_hit_format_at_{i}"
    return None

def _looks_like_system_prompt_leak(text: str) -> bool:
    low = (text or "").lower()
    leak_markers = [
        "tools allowed:",
        "you are the nasa knowledge assistant",
        "strict json format",
        "system_router",
        "system prompt",
        "developer prompt",
    ]
    return any(m in low for m in leak_markers)

class NasaAgent:
    def __init__(self):
        pass

    def run(self, user_query: str, model_mode=None, mode: str = "standard", top_k: int = 3) -> Dict[str, Any]:
        if detect_injection(user_query):
            return {
                "answer": "I cannot process this request due to safety guidelines (Injection Detected).",
                "status": "blocked_injection",
                "tool_used": None
            }
        try:
            clean_query = scrub_user_input(user_query)
        except Exception:
            clean_query = user_query.strip()
        if contains_path_traversal(clean_query):
            return {
                "answer": "Blocked: path traversal attempt.",
                "status": "blocked_path_traversal",
                "tool_used": None
            }
        low = clean_query.lower()
        if "system prompt" in low or "developer prompt" in low:
            return {
                "answer": "Blocked: cannot reveal system prompt.",
                "status": "blocked_system_prompt_request",
                "tool_used": None
            }
        if contains_profanity(clean_query):
            return {
                "answer": "I cannot process this request because it contains inappropriate language.",
                "status": "blocked_profanity",
                "tool_used": None
            }
        if links_not_allowed(clean_query):
            return {
                "answer": "I cannot process this request because it contains unauthorized links.",
                "status": "blocked_link",
                "tool_used": None
            }
        pii_flags = contains_pii(clean_query)
        if any(pii_flags.values()):
            print(f"Warning: Input contains potential PII: {pii_flags}")
        print(f"Processing Query: {clean_query}")
        router_resp = chat_once(
            prompt=f"User input: {clean_query}\n\nDecide: Return Tool JSON or Text Answer.",
            system=SYSTEM_ROUTER,
            temperature=0.0,
            max_output_tokens=200,
            model_mode=model_mode
        )
        text_out = router_resp.get("text", "")
        tool_call_obj = None
        last_err = None
        try:
            tool_call_obj = self._extract_json_tool(text_out)
        except Exception as e:
            last_err = str(e)
        if tool_call_obj is None:
            looks_like_tool = ("\"tool\"" in text_out) or ("{tool" in text_out.lower()) or (
                        "kb.lookup" in text_out) or text_out.strip().startswith("{")
            if looks_like_tool:
                repair_prompt = _make_repair_prompt(text_out)
                router_resp2 = chat_once(
                    prompt=repair_prompt,
                    system=SYSTEM_ROUTER,
                    temperature=0.0,
                    model_mode=model_mode
                )
                text_out2 = router_resp2.get("text", "")
                try:
                    tool_call_obj = self._extract_json_tool(text_out2)
                    text_out = text_out2
                except Exception as e:
                    last_err = str(e)
        if tool_call_obj is None:
            tool_call_obj = ToolCall(tool="kb.lookup", args={"query": clean_query, "top_k": top_k})
        print(f"Tool Selected: {tool_call_obj.tool} | Args: {tool_call_obj.args}")
        # dispatcher
        ok, result, err = run_tool_safe(tool_call_obj)
        if not ok:
            return {
                "answer": f"Error executing tool: {err}",
                "status": "error",
                "tool_used": tool_call_obj.tool
            }
        if tool_call_obj.tool == "kb.lookup":
            verr = _validate_kb_result(result)
            if verr:
                return {"answer": verr, "status": "error", "tool_used": "kb.lookup"}
        final_answer = self._synthesize_answer(
            clean_query,
            tool_call_obj.tool,
            result,
            model_mode=model_mode,
            mode=mode
        )
        if _looks_like_system_prompt_leak(final_answer):
            return {
                "answer": "blocked: I can’t share internal instructions or system prompts.",
                "status": "blocked_system_prompt_leak",
                "tool_used": tool_call_obj.tool
            }
        return {
            "answer": final_answer,
            "status": "ok",
            "tool_used": tool_call_obj.tool,
            "tool_result": result
        }

    def _extract_json_tool(self, text: str):
        clean = (text or "").replace("```json", "").replace("```", "").strip()
        data = _extract_json_object(clean)  # data jest dict albo None
        if not data:
            raise ValueError("validation: no_json_object_found")
        if not isinstance(data, dict) or "tool" not in data or "args" not in data:
            raise ValueError("validation: tool_json_missing_required_fields")
        try:
            return ToolCall(**data)
        except Exception as e:
            raise ValueError(f"validation: toolcall_schema_error ({e})")

    def _synthesize_answer(self, query: str, tool_name: str, result: Dict, model_mode=None, mode: str = "standard") -> str:
        m = (mode or "standard").lower()
        if m == "concise":
            style_hint = "Write a concise answer (3-6 bullet points)."
            max_toks = 260
        elif m == "detailed":
            style_hint = "Write a detailed, well-structured answer with clear sections."
            max_toks = 900
        else:
            style_hint = "Write a normal-length answer."
            max_toks = 520
        if tool_name == "kb.lookup":
            if not result.get("found"):
                return "I searched the NASA database, but I did not find an answer."

            hits = result.get("hits", [])
            context_txt = ""
            for h in hits:
                if isinstance(h, (list, tuple)):
                    doc = h[1]
                else:
                    doc = h
                content = doc.get('chunk', doc.get('content', ''))
                source = doc.get('source', 'Unknown')
                title = doc.get('title', '')
                context_txt += f"- [{title}] {content} (Source: {source})\n"

            sys_prompt = f"You are a NASA expert. Answer based ONLY on the provided context. {style_hint}"
            user_prompt = f"Question: {query}\n\nContext:\n{context_txt}\n\nAnswer in English and cite sources:"

            resp = chat_once(user_prompt, system=sys_prompt, temperature=0.0, model_mode=model_mode)
            return resp['text']

        elif "calculator" in tool_name:
            return f"Result: {result['result']}"

        elif tool_name == "units.convert":
            unit = result.get("unit", "")
            return f"{result['result']} {unit}".strip()
        else:
            return str(result)