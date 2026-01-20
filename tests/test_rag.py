from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_rag_uses_kb_lookup_tool():
    r = client.post("/ask", json={
        "query": "According to the NASA database, explain what satellites are and cite sources.",
        "use_functions": True,
        "k": 3,
        "mode": "standard",
        "model_mode": "local"
    })
    if r.status_code != 200:
        print("STATUS:", r.status_code)
        print("BODY:", r.text)
    assert r.status_code == 200

def test_rag_unknown_topic_fails_gracefully():
    r = client.post("/ask", json={
        "query": "According to the NASA database, explain faster-than-light quantum AI drive used by NASA in 2035.",
        "use_functions": True,
        "k": 3,
        "mode": "standard",
        "model_mode": "local"
    })
    assert r.status_code == 200
    data = r.json()
    ans = (data.get("answer") or "").lower()
    assert ("did not find" in ans) or ("don't know" in ans) or ("insufficient" in ans) or ("does not contain" in ans) or ("sorry" in ans) or ("no mention" in ans)
