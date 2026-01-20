from dotenv import load_dotenv
from google import genai
from google.genai import types
import os, torch, time
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

load_dotenv()

MODEL_MODE = os.getenv("MODEL_MODE", 'gemini')
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)

BASE_URL = os.getenv("GROQ_BASE_URL")
MODEL_NAME = os.getenv("GROQ_MODEL_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

_local_tokenizer = None
_local_model = None
_device = None


def get_local_resources():
    global _local_tokenizer, _local_model, _device
    if _local_model is not None:
        return _local_tokenizer, _local_model, _device
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_NAME,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        _local_tokenizer = tokenizer
        _local_model = model
        _device = device
        return tokenizer, model, device
    except Exception as e:
        print(f"Error loading local model: {e}")
        raise e


def build_prompt(tokenizer, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"[SYSTEM]\n{system}\n[USER]\n{user}\n[ASSISTANT]\n"


def count_tokens_local(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


gclient = None
if GOOGLE_API_KEY and genai:
    gclient = genai.Client(api_key=GOOGLE_API_KEY)

client = None
if GROQ_API_KEY:
    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")


def _is_rate_limited(err: Exception) -> bool:
    msg = str(err).lower()
    return (
            "resource_exhausted" in msg
            or "quota" in msg
            or "429" in msg
            or "rate limit" in msg
            or "too many requests" in msg
    )


def _looks_like_gemini_rate_limit(out: dict) -> bool:
    txt = (out or {}).get("text", "")
    low = txt.lower()
    return (
            "error gemini" in low
            and (
                    "resource_exhausted" in low
                    or "quota" in low
                    or "429" in low
                    or "overloaded" in low
            )
    )


@torch.inference_mode()
def local_generate(
        prompt: str,
        system: str = "You are a helpful assistant.",
        max_output_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
) -> Dict[str, Any]:
    tokenizer, model, device = get_local_resources()

    t0 = time.perf_counter()
    text = build_prompt(tokenizer, system, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    do_sample = temperature > 0.0
    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_output_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)

    output_ids = model.generate(**inputs, **gen_kwargs)
    gen_only = output_ids[0, inputs["input_ids"].shape[-1]:]
    output_txt = tokenizer.decode(gen_only, skip_special_tokens=True)
    dt = time.perf_counter() - t0

    ptoks = count_tokens_local(tokenizer, prompt) + count_tokens_local(tokenizer, system)
    ctoks = count_tokens_local(tokenizer, output_txt)

    return {
        "text": output_txt,
        "latency_s": round(dt, 3),
        "usage": {
            "prompt_tokens": ptoks,
            "completion_tokens": ctoks,
            "total_tokens": ptoks + ctoks,
        }
    }


def gemini_generate(prompt: str, system: str = "You are a helpful assistant", temperature: float = 0.0,
                    top_p: float = 1.0, top_k: int = 40, max_output_tokens: int = 256) -> dict:
    if not gclient:
        return {"text": "Error: Google API Key missing.", "usage": {}}
    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            response = gclient.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_output_tokens=max_output_tokens
                ),
            )
            dt = time.perf_counter() - t0
            usage = getattr(response, "usage_metadata", None)
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "completion_tokens": getattr(usage, "candidates_token_count", None),
                "total_tokens": getattr(usage, "total_token_count", None),
            } if usage is not None else None
            text = getattr(response, "text", None)
            return {
                "text": text if text is not None else str(response),
                "latency_s": round(dt, 3),
                "usage": usage_dict,
            }
        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "429" in error_str or "overloaded" in error_str.lower():
                print(f"Gemini Overloaded (Attempt {attempt + 1}/{max_retries}). Retrying in 2s...")
                time.sleep(2 * (attempt + 1))
                last_error = e
                continue
            else:
                return {"text": f"Error Gemini: {e}", "usage": {}}
    return {"text": f"Error Gemini (Max Retries Exceeded): {last_error}", "usage": {}}


def groq_generate(prompt: str, system: str = "You are a helpful assistant", temperature: float = 0.0,
                  top_p: float = 1.0, max_output_tokens: int = 256) -> dict:
    if client is None:
        raise RuntimeError("Groq client not configured (missing GROQ_API_KEY).")
    t0 = time.perf_counter()
    response = client.responses.create(
        model=MODEL_NAME,
        instructions=system,
        input=prompt,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
    )
    dt = time.perf_counter() - t0
    usage = getattr(response, "usage", None)
    usage_dict = None if usage is None else getattr(usage, "model_dump", lambda: usage)()
    return {
        "text": response.output_text,
        "latency_s": round(dt, 3),
        "usage": usage_dict,
    }


def chat_once(
        prompt: str,
        system: str = "You are a helpful assistant.",
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        max_output_tokens: int = 256,
        model_mode: str | None = None,
) -> dict:
    mode = (model_mode or os.getenv("MODEL_MODE", "gemini") or "local").lower()

    if mode == "gemini":
        out = gemini_generate(prompt, system, temperature, top_p, top_k or 40, max_output_tokens)
        if _looks_like_gemini_rate_limit(out):
            try:
                out2 = groq_generate(prompt, system, temperature, top_p, max_output_tokens)
                out2["fallback_used"] = "groq"
                return out2
            except Exception:
                out3 = local_generate(prompt, system, max_output_tokens, temperature, top_p, top_k)
                out3["fallback_used"] = "local"
                return out3
        return out

    if mode == "groq":
        try:
            return groq_generate(prompt, system, temperature, top_p, max_output_tokens)
        except Exception:
            return local_generate(prompt, system, max_output_tokens, temperature, top_p, top_k)

    return local_generate(prompt, system, max_output_tokens, temperature, top_p, top_k)