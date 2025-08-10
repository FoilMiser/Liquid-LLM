
# scripts/make_self_instruct.py
"""
Generate instruction/response pairs with the teacher model.
Save JSONL with fields: {"prompt": ..., "response": ...}
"""
import json, os, random

def ask_teacher(prompt):
    # TODO: Replace with real teacher call (DeepSeek-R1-0528)
    return "This is a placeholder response. Replace with teacher output."

def main(out_path="data/sft/self_instruct_2k.jsonl", n=2000):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seeds = [
        "Explain the difference between supervised and unsupervised learning.",
        "Write a short poem about rain and memory.",
        "Summarize the causes of the French Revolution in 5 bullet points.",
        "Give 3 examples of everyday Bayesian reasoning.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n):
            prompt = random.choice(seeds)
            resp = ask_teacher(prompt)
            f.write(json.dumps({"prompt": prompt, "response": resp}, ensure_ascii=False) + "\n")
    print(f"Wrote {n} pairs to {out_path}")

if __name__ == "__main__":
    main()
