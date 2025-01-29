import torch
from transformers import pipeline
from modelscope import snapshot_download

model_dir = snapshot_download('LLM-Research/Llama-3.2-1B-Instruct')

pipe = pipeline(
    "text-generation",
    model=model_dir,
    torch_dtype=torch.bfloat16,
    #device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])