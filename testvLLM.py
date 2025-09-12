import os
# Imposta cache locali dentro il progetto
os.environ["HF_HOME"] = "/workspace/gerlando/.cache_hf"
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["XDG_CACHE_HOME"] = os.environ["HF_HOME"]
os.environ["UV_PYTHON_CACHE"] = "/workspace/gerlando/.cache_uv"

import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
import zipfile
from io import TextIOWrapper

def count_tokens(tokenizer, text, add_special_tokens=False):
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))

def main():
    parser = argparse.ArgumentParser(description="Generate LLM outputs per tag from a zip of prompts.")
    parser.add_argument("--zip_path", type=str, default="/workspace/gerlando/TestCCO/prompts_per_task_v06.zip", help="Path to the zip containing the JSON prompts")
    parser.add_argument("--json_name", type=str, default="prompts_per_task_v06.json", help="Name of the JSON file inside the zip")
    parser.add_argument("--start_tag", type=int, default=0, help="Index of the first tag to process (inclusive)")
    parser.add_argument("--end_tag", type=int, default=None, help="Index of the last tag to process (exclusive)")
    parser.add_argument("--tags", type=str, nargs="*", help="Optional list of tag names to process (overrides start/end indices)")
    parser.add_argument("--output_dir", type=str, default="/workspace/gerlando/TestCCO/results", help="Directory to save output JSON files")
    args = parser.parse_args()

    # Initialize model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    llm = LLM(model=model_name, max_model_len=50000)

    # Load JSON from zip
    with zipfile.ZipFile(args.zip_path, 'r') as zf:
        with zf.open(args.json_name) as f:
            prompts = json.load(TextIOWrapper(f, encoding='utf-8'))

    # Determine which tags to process
    if args.tags:
        tag_list = [tag for tag in args.tags if tag in prompts]
    else:
        end_tag = args.end_tag if args.end_tag is not None else len(prompts)
        tag_list = list(prompts.keys())[args.start_tag:end_tag]

    max_out = 10000
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_out)

    for tag in tag_list:
        tag_prompts = prompts[tag]
        print(f"Generating for tag: {tag} ({len(tag_prompts)} prompts)")
        
        outputs = llm.generate(tag_prompts, sampling_params)

        results = []
        for i, gen_result in enumerate(outputs):
            results.append({
                "prompt": tag_prompts[i],
                "generated_text": [c.text for c in gen_result.outputs]
            })

        output_file = f"{args.output_dir}/{tag}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Saved results for tag '{tag}' in {output_file}")

if __name__ == "__main__":
    main()
