import os
import json
import requests

# Default Judge Model Key (Matches agents.json shorthand)
DEFAULT_JUDGE_MODEL_KEY = "gpt5nano"

def resolve_judge_model(model_key, agents_json_path=None):
    """Resolves model shorthand against agents.json in core/configs/."""
    if agents_json_path is None:
        # Resolve relative to this file's position (core/judge_core/ -> core/configs/agents.json)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        agents_json_path = os.path.join(base_dir, "configs", "agents.json")
        
        # Fallback to root or current working dir if not found in core/configs
        if not os.path.exists(agents_json_path):
            agents_json_path = "core/configs/agents.json" if os.path.exists("core/configs/agents.json") else "agents.json"

    if os.path.exists(agents_json_path):
        try:
            with open(agents_json_path, "r") as f:
                agents = json.load(f)
            return agents.get(model_key, model_key)
        except Exception as e:
            print(f"[Judge Config Warning] Failed to parse {agents_json_path}: {e}")
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
    
    # Optional debug print if another bad request happens
    if not response.ok:
        print(f"[OpenRouter Error Response]: {response.text}")
        
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]
