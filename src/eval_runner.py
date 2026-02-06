#!/usr/bin/env python3
"""
Norwegian LLM Evaluation Framework
Compares Norwegian-tuned models vs International baselines

Test questions are loaded from external JSONL file (not included in public repo).
See data/golden_sets/example_tests.jsonl for format.
"""

import requests
import json
import argparse
from datetime import datetime
from pathlib import Path


def load_tests(filepath: str) -> list:
    """Load test questions from JSONL file."""
    tests = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tests.append(json.loads(line))
    return tests


def query_ollama(model_id: str, prompt: str, timeout: int = 300) -> str:
    """Query Ollama via HTTP API with token limit to prevent loops."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_id,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 250,
                    "temperature": 0.7
                }
            },
            timeout=timeout
        )
        data = response.json()
        return data.get("response", "[NO RESPONSE]")
    except requests.exceptions.Timeout:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"


def run_evaluation(tests: list, models: list, output_prefix: str = "eval_results"):
    """Run evaluation across all tests and models."""
    
    print("=" * 60)
    print("Norwegian LLM Evaluation Framework")
    print("=" * 60)
    print(f"Tests: {len(tests)}")
    print(f"Models: {[m[0] for m in models]}")
    print(f"Total queries: {len(tests) * len(models)}")
    print("=" * 60)
    print()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "models": [m[0] for m in models],
        "tests": []
    }
    
    total = len(tests) * len(models)
    current = 0
    
    for test in tests:
        test_result = {
            "id": test.get("id", "unknown"),
            "question": test.get("question") or test.get("q"),
            "category": test.get("category", "general"),
            "responses": {}
        }
        
        for model_name, model_id in models:
            current += 1
            print(f"[{current}/{total}] {test_result['id']}: {model_name}...", end=" ", flush=True)
            
            response = query_ollama(model_id, test_result["question"])
            test_result["responses"][model_name] = response
            print("done")
        
        # Save after each question (crash-proof)
        results["tests"].append(test_result)
        with open(f"{output_prefix}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print()
    
    # Generate CSV
    print("\nGenerating CSV...")
    model_names = [m[0] for m in models]
    
    with open(f"{output_prefix}.csv", "w", encoding="utf-8") as f:
        # Header (tab-separated for Excel compatibility)
        header = ["ID", "Category", "Question"] + model_names
        f.write("\t".join(header) + "\n")
        
        # Rows
        for test in results["tests"]:
            row = [
                test["id"],
                test["category"],
                test["question"].replace("\t", " ").replace("\n", " ")
            ]
            
            for name in model_names:
                resp = test["responses"].get(name, "")
                resp = resp.replace("\t", " ").replace("\n", " ")[:500]
                row.append(resp)
            
            f.write("\t".join(row) + "\n")
    
    print(f"\n{'=' * 60}")
    print(f"✓ Results saved to {output_prefix}.json")
    print(f"✓ CSV saved to {output_prefix}.csv (tab-separated)")
    print(f"  Tested {len(tests)} questions × {len(models)} models = {total} responses")
    print(f"{'=' * 60}")
    
    return results


# Default model configurations
NORWEGIAN_MODELS = [
    ("NB-Llama-3.2-3B", "hf.co/NbAiLab/nb-llama-3.2-3B-Q4_K_M-GGUF:latest"),
]

INTERNATIONAL_MODELS = [
    ("Llama-3.2-3B", "llama3.2:3b"),
    ("Mistral-7B", "mistral:7b"),
    ("Llama-3.1-8B", "llama3.1:8b"),
]


def main():
    parser = argparse.ArgumentParser(description="Norwegian LLM Evaluation Framework")
    parser.add_argument(
        "--tests", "-t",
        required=True,
        help="Path to JSONL file containing test questions"
    )
    parser.add_argument(
        "--output", "-o",
        default="eval_results",
        help="Output file prefix (default: eval_results)"
    )
    parser.add_argument(
        "--norwegian-only",
        action="store_true",
        help="Only test Norwegian models"
    )
    parser.add_argument(
        "--international-only",
        action="store_true",
        help="Only test international models"
    )
    
    args = parser.parse_args()
    
    # Select models
    if args.norwegian_only:
        models = NORWEGIAN_MODELS
    elif args.international_only:
        models = INTERNATIONAL_MODELS
    else:
        models = NORWEGIAN_MODELS + INTERNATIONAL_MODELS
    
    # Load and run
    tests = load_tests(args.tests)
    run_evaluation(tests, models, args.output)


if __name__ == "__main__":
    main()
