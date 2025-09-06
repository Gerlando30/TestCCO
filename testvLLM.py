from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)  # trust_remote_code utile per tokenizers custom
llm = LLM(model=model_name, max_model_len=70000)

prompts = json.load(open("/root/TestCCO/random_prompts.json"))
only_prompts = [p for tag in prompts for p in prompts[tag]]

def count_tokens(text, add_special_tokens=False):
    # encode() restituisce una lista di token ids; la lunghezza è il conteggio dei token
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))

max_model_len = 70000
safety_margin = 8  # lascia qualche token per BOS/EOS / special tokens

results = []
for prompt in only_prompts:
    in_toks = count_tokens(prompt, add_special_tokens=False)
    max_out = max_model_len - in_toks - safety_margin
    if max_out <= 0:
        max_out = 16  # minimo sicuro se l'input è troppo lungo
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_out)

    outputs = llm.generate([prompt], sampling_params)
    gen = outputs[0].outputs[0].text
    results.append({"prompt": prompt, "in_tokens": in_toks, "max_output_tokens": max_out, "generated_text": gen})
    print(f"in={in_toks} out_max={max_out}\n{gen}\n{'-'*60}")

with open("/root/TestCCO/generated_outputs.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
