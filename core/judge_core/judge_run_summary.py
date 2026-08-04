import os
import json
import random
import argparse
import numpy as np

from core.configs import rubric_config
from core.judge_core.judge_config import resolve_judge_model, call_judge_llm

def summarize_run(run_dir, judge_model_key, api_key):
    trials_dir = os.path.join(run_dir, "trials")
    if not os.path.exists(trials_dir):
        print(f"[Judge Summary Error] Trials directory not found at {trials_dir}")
        return

    judge_files = []
    for root, _, files in os.walk(run_dir):
        for file in files:
            if file == "trial_judge.json":
                judge_files.append(os.path.join(root, file))

    if not judge_files:
        print(f"[Judge Summary Error] No trial_judge.json files found in {run_dir}")
        return

    # 1. Exact Mathematical Aggregation across ALL Trials
    all_final_scores = []
    category_score_lists = {cat: [] for cat in rubric_config.RUBRIC_CATEGORIES}
    checkpoint_score_lists = {}

    for jf in judge_files:
        with open(jf, "r") as f:
            data = json.load(f)

        all_final_scores.append(data["final_weighted_score"])
        
        for cat, score in data.get("category_scores", {}).items():
            if cat in category_score_lists:
                category_score_lists[cat].append(score)

        for chk, details in data.get("checkpoint_details", {}).items():
            if chk not in checkpoint_score_lists:
                checkpoint_score_lists[chk] = []
            checkpoint_score_lists[chk].append(details["points"])

    mean_final_score = float(np.mean(all_final_scores))
    mean_category_scores = {cat: float(np.mean(scores)) for cat, scores in category_score_lists.items() if scores}
    mean_checkpoint_scores = {chk: float(np.mean(scores)) for chk, scores in checkpoint_score_lists.items() if scores}

    # 2. Random Sampling (Up to 20 files for qualitative meta-analysis)
    sampled_files = random.sample(judge_files, min(len(judge_files), 20))
    sampled_summaries = []
    for sf in sampled_files:
        with open(sf, "r") as f:
            d = json.load(f)
            sampled_summaries.append({
                "file": os.path.basename(os.path.dirname(sf)),
                "score": d["final_weighted_score"],
                "summary": d.get("qualitative_summary", {})
            })

    # 3. Meta-Analysis LLM Call
    model_id = resolve_judge_model(judge_model_key)
    system_prompt = """You are a senior benchmark evaluator synthesizing overall LLM performance across an entire experimental run.
Provide a clear, high-level qualitative summary of macro strengths, system failure patterns, and recommendations. Return valid JSON."""

    user_prompt = f"""
GLOBAL MATHEMATICAL AVERAGES (N={len(judge_files)} trials):
- Global Final Score Mean: {mean_final_score:.4f}
- Category Averages: {json.dumps(mean_category_scores, indent=2)}

SAMPLED QUALITATIVE TRIAL SUMMARIES (Sample of {len(sampled_files)}):
{json.dumps(sampled_summaries, indent=2)}

REQUIRED JSON OUTPUT FORMAT:
{{
  "macro_successes": "...",
  "macro_failure_modes": "...",
  "recommendations_for_agent_improvement": "..."
}}
"""

    raw_response = call_judge_llm(system_prompt, user_prompt, model_id, api_key, response_format_json=True)
    meta_summary = json.loads(raw_response)

    run_judge_output = {
        "rubric_version": rubric_config.RUBRIC_VERSION,
        "rubric_hash": rubric_config.get_rubric_hash(),
        "total_trials_evaluated": len(judge_files),
        "judge_model": model_id,
        "global_scores": {
            "mean_final_weighted_score": round(mean_final_score, 4),
            "mean_category_scores": {k: round(v, 4) for k, v in mean_category_scores.items()},
            "mean_checkpoint_scores": {k: round(v, 4) for k, v in mean_checkpoint_scores.items()}
        },
        "qualitative_meta_analysis": meta_summary
    }

    output_path = os.path.join(run_dir, "run_judge_summary.json")
    with open(output_path, "w") as f:
        json.dump(run_judge_output, f, indent=4)

    print(f"\n{'='*70}\n[Judge Core] Overall Run Summary saved -> {output_path}\n{'='*70}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default="claude-3-5")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    summarize_run(args.run_dir, args.judge_model, api_key)
