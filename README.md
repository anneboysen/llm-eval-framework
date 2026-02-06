# Norwegian LLM Evaluation Framework

A framework for evaluating Large Language Models on Norwegian language tasks, comparing Norwegian-tuned models against international baselines.

## What This Tests

The framework evaluates models across failure categories particularly relevant for Norwegian:

- Compound words - Norwegian compound formation and recognition
- Offensive language - Culturally-specific slurs and offensive terms  
- Safety/PII - Handling of personnummer and other sensitive data
- Cultural knowledge - Norwegian history, mythology, local knowledge
- False friends - Swedish/Danish words that differ in Norwegian
- Sami language - Recognition of Sami loanwords
- Disability language - Appropriate terminology and framing

## Installation

Clone the repository and install dependencies:

    git clone https://github.com/anneboysen/llm-eval-framework.git
    cd llm-eval-framework
    pip install requests

Install Ollama for local model testing: https://ollama.ai

## Usage

Prepare a JSONL file with test questions:

    {"id": "TEST-001", "question": "Your question here", "category": "category_name"}
    {"id": "TEST-002", "question": "Another question", "category": "category_name"}

Pull models:

    ollama pull hf.co/NbAiLab/nb-llama-3.2-3B-Q4_K_M-GGUF:latest
    ollama pull llama3.2:3b
    ollama pull mistral:7b

Run evaluation:

    python src/eval_runner.py --tests path/to/your/tests.jsonl

Results save to eval_results.json and eval_results.csv.

## Test Questions

The full test suite is proprietary and not included in this repository. The framework works with your own test questions - see data/golden_sets/example_tests.jsonl for the expected format.

For evaluation services or access to the Norwegian test suite, contact me on LinkedIn.

## Key Findings

Our evaluation of Norwegian "sovereign AI" models revealed systematic failures in compound slur recognition (models hallucinate definitions instead of flagging offensive content), PII safety (some models attempt to complete partial national ID numbers), and cultural context.

## Author

Anne Boysen - AI Evaluation & Strategic Operations
https://linkedin.com/in/anneboysen
