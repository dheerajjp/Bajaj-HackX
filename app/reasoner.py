import json, os
from typing import List, Dict, Any
from .config import OPENAI_API_KEY

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

def _prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    ctx_text = "\n\n".join([f"[Chunk {c['id']} | score={c['score']:.2f}]\n{c['text']}" for c in contexts])
    template = f"""
You are a domain expert (insurance/legal/HR/compliance). Answer the QUESTION using only the CONTEXT.
If the answer is not explicitly supported, say "Not explicitly stated." Be concise.

Return STRICT JSON with keys: answer (string), reasoning (string, <= 2 sentences), confidence (0..1), citations (array of chunk ids).

CONTEXT:
{ctx_text}

QUESTION:
{question}

JSON ONLY:
""".strip()
    return template

def _parse_json(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except Exception:
        import re
        m = re.search(r'\{.*\}', content, re.S)
        if m:
            return json.loads(m.group(0))
        return {"answer": content, "reasoning": "LLM returned free text.", "confidence": 0.5, "citations": []}

def call_llm(question: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = _prompt(question, contexts)

    # Dev/heuristic mode if no key
    if not OPENAI_API_KEY:
        text = contexts[0]['text'] if contexts else ""
        ans = "Not explicitly stated."
        if any(k in text.lower() for k in ["cover", "covered", "coverage"]):
            ans = "Possibly covered; conditions may apply. Refer to cited clauses."
        return {
            "answer": ans,
            "reasoning": "OpenAI key not set; returned heuristic answer from local mode.",
            "confidence": 0.4,
            "citations": [c["id"] for c in contexts[:2]]
        }

    # New SDK only
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You reason carefully and cite sources by chunk id."},
                {"role": "user", "content": prompt}
            ]
        )
        content = resp.choices[0].message.content.strip()
        data = _parse_json(content)
    except Exception as e:
        return {
            "answer": "Not explicitly stated.",
            "reasoning": f"LLM error: {e}",
            "confidence": 0.3,
            "citations": [c["id"] for c in contexts[:1]]
        }

    # Safety defaults
    data.setdefault("answer", "Not explicitly stated.")
    data.setdefault("reasoning", "")
    data.setdefault("confidence", 0.6)
    data.setdefault("citations", [])
    return data
