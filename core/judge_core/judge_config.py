import os
import json
import requests

# Default Judge Model Key (Matches agents.json shorthand)
DEFAULT_JUDGE_MODEL_KEY = "claude-3-5"

def resolve_judge_model(model_key, agents_json_path="agents.json"):
    """Resolves model shorthand against agents.json."""
    if os.path.exists(agents_json_path):
        try:
            with open(agents_json_path, "r") as f:
                agents = json.load(f)
            return agents.get(model_key, model_key)
        except Exception:
            pass
    return model_key

def call_judge_llm(system_prompt, user_prompt, model_id, api_key, response_format_json=True):
    """Executes a synchronous, low-temperature LLM call to OpenRouter."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_id,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]
