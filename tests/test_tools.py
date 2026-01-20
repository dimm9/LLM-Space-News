from app.tools.registry import ToolCall
from app.tools.handlers import run_tool_safe

print("=== Test Calculator ===")
tc_calc = ToolCall(tool="calculator.add", args={"a": 5, "b": 10})
ok, res, err = run_tool_safe(tc_calc)
print(f"Status: {ok}, Result: {res}")

print("\n=== Test RAG ===")
tc_rag = ToolCall(tool="kb.lookup", args={"query": "Electron rocket", "top_k": 2})
ok, res, err = run_tool_safe(tc_rag)
print(f"Status: {ok}")
if ok:
    print(f"Found: {len(res.get('hits', []))} documents.")
else:
    print(f"Error: {err}")