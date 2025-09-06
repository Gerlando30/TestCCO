from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import json

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)  # trust_remote_code utile per tokenizers custom
llm = LLM(model=model_name, max_model_len=50000)

prompts = json.load(open("/root/TestCCO/random_prompts.json"))
only_prompts = [p for tag in prompts for p in prompts[tag]]

def count_tokens(text, add_special_tokens=False):
    # encode() restituisce una lista di token ids; la lunghezza è il conteggio dei token
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))

max_model_len = 70000
safety_margin = 8  # lascia qualche token per BOS/EOS / special tokens
# json_schema = {
#   "type": "array",
#   "minItems": 3, "maxItems": 3,
#   "items": {
#     "type": "object",
#     "properties": {
#       "model_name": {"type": "string"},
#       "confidence": {"type": "number", "minimum": 0, "maximum": 1}
#     },
#     "required": ["model_name","confidence"]
#   }
# }
# guided = GuidedDecodingParams(json=json_schema)

max_out = 10000
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_out) #, guided_decoding=guided)
outputs = llm.generate(only_prompts, sampling_params)



results = []
for i, gen_result in enumerate(outputs):
    results.append({
        "prompt": only_prompts[i],
        "generated_text": [c.text for c in gen_result.outputs]
    })

# Salva su file
with open("/root/TestCCO/generated_outputs.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)





# for prompt in only_prompts[:10]:
#     in_toks = count_tokens(prompt, add_special_tokens=False)
#     max_out = max_model_len - in_toks - safety_margin
#     if max_out <= 0:
#         max_out = 16  # minimo sicuro se l'input è troppo lungo
#     sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_out) #, guided_decoding=guided)

#     outputs = llm.generate([prompt], sampling_params)
#     gen = outputs[0].outputs[0].text
#     results.append({"prompt": prompt, "in_tokens": in_toks, "max_output_tokens": max_out, "generated_text": gen})
#     print(f"in={in_toks} out_max={max_out}\n{gen}\n{'-'*60}")

# with open("/root/TestCCO/generated_outputs.json", "w", encoding="utf-8") as f:
#     json.dump(results, f, ensure_ascii=False, indent=2)
