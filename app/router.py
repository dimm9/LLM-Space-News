import json
import logging
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
    links_not_allowed,
    contains_path_traversal,
    ALLOWED_DOMAINS
)
logger = logging.getLogger(__name__)

METRICS = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "tool_calls": 0,
    "security_blocks": 0
}

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
- Always include at least one 'Source:' line. It should be from provided context.
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _fallback_kb_toolcall(user_query: str, top_k: int = 3) -> ToolCall:
    return ToolCall(tool="kb.lookup", args={"query": user_query, "top_k": top_k})


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text: return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict): return obj
    except Exception:
        pass
    for m in re.findall(r"\{.*?\}", text, re.DOTALL):
        try:
            obj = json.loads(m)
            if isinstance(obj, dict): return obj
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


def _validate_kb_result(result: Dict[str, Any]) -> str | None:
    if not isinstance(result, dict): return "validation: invalid_kb_response_not_dict"
    if result.get("error"): return f"validation: kb_error_returned ({result.get('error')})"
    if result.get("found") is False: return None
    hits = result.get("hits")
    if not isinstance(hits, list): return "validation: invalid_kb_response_hits_not_list"
    return None


def _looks_like_system_prompt_leak(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in
               ["tools allowed:", "you are the nasa knowledge assistant", "strict json format", "system_router",
                "system prompt"])


class NasaAgent:
    def __init__(self):
        pass

    def run(self, user_query: str, model_mode=None, mode: str = "standard", top_k: int = 3) -> Dict[str, Any]:
        METRICS["total_requests"] += 1

        if detect_injection(user_query):
            return {"answer": "I cannot process this request (Injection Detected).", "status": "blocked_injection",
                    "tool_used": None}
        try:
            clean_query = scrub_user_input(user_query)
        except Exception:
            clean_query = user_query.strip()

        if contains_path_traversal(clean_query):
            METRICS["security_blocks"] += 1
            METRICS["failed_requests"] += 1
            return {"answer": "Blocked: path traversal attempt.", "status": "blocked_path_traversal", "tool_used": None}
        if contains_profanity(clean_query):
            METRICS["security_blocks"] += 1
            METRICS["failed_requests"] += 1
            return {"answer": "Blocked: inappropriate language.", "status": "blocked_profanity", "tool_used": None}
        pii_detected = contains_pii(clean_query)
        if any(pii_detected.values()):
            METRICS["security_blocks"] += 1
            METRICS["failed_requests"] += 1
            return {"answer": "Blocked: PII detected in query.", "status": "blocked_pii", "tool_used": None}
        if links_not_allowed(clean_query):
            METRICS["security_blocks"] += 1
            METRICS["failed_requests"] += 1
            return {"answer": "Blocked: Query contains unauthorized links.", "status": "blocked_links",
                    "tool_used": None}
        logger.info(f"Processing Query: {clean_query}")

        router_resp = chat_once(
            prompt=f"User input: {clean_query}\n\nDecide: Return Tool JSON or Text Answer.",
            system=SYSTEM_ROUTER,
            temperature=0.0,
            max_output_tokens=200,
            model_mode=model_mode
        )
        text_out = router_resp.get("text", "")
        tool_call_obj = None
        try:
            tool_call_obj = self._extract_json_tool(text_out)
        except Exception:
            # fallback
            looks_like_tool = ("\"tool\"" in text_out) or ("{tool" in text_out.lower()) or ("kb.lookup" in text_out)
            if looks_like_tool:
                logger.info("Invalid JSON received, attempting repair...")
                router_resp2 = chat_once(prompt=_make_repair_prompt(text_out), system=SYSTEM_ROUTER, temperature=0.0,
                                         model_mode=model_mode)
                try:
                    tool_call_obj = self._extract_json_tool(router_resp2.get("text", ""))
                except Exception:
                    pass

        if tool_call_obj is None:
            logger.warning("No tool selected by LLM, defaulting to kb.lookup")
            tool_call_obj = ToolCall(tool="kb.lookup", args={"query": clean_query, "top_k": top_k})
        METRICS["tool_calls"] += 1
        logger.info(f"Tool Selected: {tool_call_obj.tool} | Args: {tool_call_obj.args}")
        ok, result, err = run_tool_safe(tool_call_obj)
        if not ok:
            logger.error(f"Tool execution error: {err}")
            return {"answer": f"Error executing tool: {err}", "status": "ok", "tool_used": tool_call_obj.tool}

        if tool_call_obj.tool == "kb.lookup":
            verr = _validate_kb_result(result)
            if verr:
                logger.warning(f"KB Validation warning: {verr}. Treating as empty result.")
                result = {"found": False, "hits": []}
        final_answer = self._synthesize_answer(clean_query, tool_call_obj.tool, result, model_mode=model_mode,
                                               mode=mode)
        if _looks_like_system_prompt_leak(final_answer):
            METRICS["security_blocks"] += 1
            METRICS["failed_requests"] += 1
            return {"answer": "Blocked: System prompt leak.", "status": "blocked", "tool_used": tool_call_obj.tool}
        METRICS["successful_requests"] += 1

        return {"answer": final_answer, "status": "ok", "tool_used": tool_call_obj.tool, "tool_result": result}

    def _extract_json_tool(self, text: str):
        clean = (text or "").replace("```json", "").replace("```", "").strip()
        data = _extract_json_object(clean)
        if not data or "tool" not in data or "args" not in data:
            raise ValueError("Invalid tool format")
        return ToolCall(**data)

    def _synthesize_answer(self, query: str, tool_name: str, result: Dict, model_mode=None,
                           mode: str = "standard") -> str:
        if tool_name == "kb.lookup":
            if not result.get("found"):
                return "I searched the NASA database, but I did not find an answer."

            hits = result.get("hits", [])
            context_txt = ""
            for h in hits:
                doc = h[1] if isinstance(h, (list, tuple)) else h
                context_txt += f"- [{doc.get('title', '')}] {doc.get('chunk', '')} (Source: {doc.get('source', '')})\n"

            style_hint = "Be concise." if mode == "concise" else "Be detailed."
            allowed_domains_str = ", ".join(ALLOWED_DOMAINS)
            sys_prompt = (
                f"You are a NASA expert. {style_hint}\n"
                f"Allowed/Trusted domains: {allowed_domains_str}\n"
                "Answer the user's question based ONLY on the provided Context below if question about topics in provided context or use information from allowed links but still list the source.\n"
                "If the answer is not about topics in the context, say 'I do not have enough information'.\n"
                "If this is a casual greeting tell that you can do (say that you are helpful space astronomy assistant).\n"
                "Do NOT use your own knowledge. Do NOT make up facts.\n"
                "Help and correct user based on provided context and topics.\n"
                "ALWAYS cite the source provided in the context or from allowed links (e.g., [Source: link from context])."
            )
            user_prompt = f"Question: {query}\n\nContext:\n{context_txt}\n\nAnswer and cite sources:"

            resp = chat_once(user_prompt, system=sys_prompt, temperature=0.0, model_mode=model_mode)
            return resp['text']

        elif "calculator" in tool_name:
            return f"Result: {result['result']}"
        elif tool_name == "units.convert":
            return f"{result['result']} {result.get('unit', '')}"
        else:
            return str(result)