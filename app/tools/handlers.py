import glob, os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, Any, Tuple, Optional

from app.tools.registry import ALLOWED_TOOLS, ToolCall, RagArgs, CalcArgs, SearchArgs, ConvertArgs
from app.rag.retriever import NasaRetriever

try:
    _RETRIEVER = NasaRetriever()
except Exception as e:
    print(f"Warning: Failed to load RAG in handlers: {e}")
    _RETRIEVER = None

DATA_DIR = os.path.abspath(os.getenv("DATA_DIR", "./data"))

def _run_tool_sync(tc: ToolCall) -> Dict[str, Any]:
    if tc.tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool not allowed {tc.tool}")
    if tc.tool.startswith("calculator."):
        args = CalcArgs(**tc.args)
        op = tc.tool.split(".")[1]
        if op == "add":
            res = args.a + args.b
        elif op == "sub":
            res = args.a - args.b
        elif op == "mul":
            res = args.a * args.b
        elif op == "div":
            if args.b == 0: raise ValueError("Division by zero")
            res = args.a / args.b
        else:
            raise ValueError("Unknown calc op")
        return {"result": res}

    if tc.tool == "units.convert":
        args = ConvertArgs(**tc.args)
        v, fr, to = args.value, args.from_unit, args.to_unit
        if fr == to:
            return {"result": round(v, 2), "unit": to}
        if fr == "km" and to == "mi":
            return {"result": round(v / 1.609344, 2), "unit": "mi"}
        if fr == "mi" and to == "km":
            return {"result": round(v * 1.609344, 2), "unit": "km"}
        if fr == "c" and to == "f":
            return {"result": round(v * 9.0 / 5.0 + 32.0, 2), "unit": "f"}
        if fr == "f" and to == "c":
            return {"result": round((v - 32.0) * 5.0 / 9.0, 2), "unit": "c"}
        raise ValueError("validation: unsupported_conversion")

    if tc.tool == "files.search":
        args = SearchArgs(**tc.args)
        raw = args.pattern[:64].strip()
        if not raw:
            raise ValueError("validation: pattern cannot be empty")
        if os.path.isabs(raw) or raw.startswith("~"):
            raise ValueError("security_blocked: absolute/tilde paths not allowed")
        raw = raw.replace("..", "")
        glob_pat = os.path.join(DATA_DIR, raw)
        out = []
        base = DATA_DIR + os.sep
        for p in glob.glob(glob_pat, recursive=True):
            rp = os.path.realpath(p)
            if not rp.startswith(base):
                continue
            if os.path.isfile(rp):
                out.append(os.path.relpath(rp, DATA_DIR))
        return {"files": out[:50]}

    if tc.tool == "kb.lookup":
        args = RagArgs(**tc.args)
        if _RETRIEVER is None:
            return {"error": "RAG system not initialized"}
        hits = _RETRIEVER.search(args.query, top_k=args.top_k)
        if not hits:
            return {"found": False, "hits": []}
        return {"found": True, "hits": hits}

    raise ValueError(f"Unhandled tool implementation: {tc.tool}")

def run_tool_safe(tc: ToolCall, timeout_s: float = 3.0) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_run_tool_sync, tc)
            result = fut.result(timeout=timeout_s)
            return True, result, None

    except TimeoutError:
        return False, {}, "Tool execution timed out (security limit)"
    except Exception as e:
        return False, {}, str(e)