from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LocoreMind/LocoOperator-4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# real sample
messages = [
    {
        "role": "system",
        "content": "You are a read-only codebase search specialist.\n\nCRITICAL CONSTRAINTS:\n1. STRICTLY READ-ONLY: You cannot create, edit, delete, move files, or run any state-changing commands. Use tools/bash ONLY for reading (e.g., ls, find, cat, grep).\n2. EFFICIENCY: Spawn multiple parallel tool calls for faster searching.\n3. OUTPUT RULES: \n   - ALWAYS use absolute file paths.\n   - STRICTLY NO EMOJIS in your response.\n   - Output your final report directly. Do not use colons before tool calls.\n\nENV: Working directory is /Users/developer/workspace/code-analyzer (macOS, zsh)."
    },
    {
        "role": "user",
        "content": "PROJECT RULES:\n- Always use `uv` for Python operations (e.g., `uv run python`, `uv pip install`).\n- `scripts/` directory is for execution only; no data files allowed.\n- Assume today is 2026-02-16.\n\nTASK:\nAnalyze the Black codebase at `/Users/developer/workspace/code-analyzer/projects/black`.\nFind and explain:\n1. How Black discovers config files.\n2. The exact search order for config files.\n3. Supported config file formats.\n4. Where this configuration discovery logic lives in the codebase.\n\nReturn a comprehensive answer explaining the exact order, including relevant code snippets and absolute file paths."
    }
]

# prepare the model input
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)