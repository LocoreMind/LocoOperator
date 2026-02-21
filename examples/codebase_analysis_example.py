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
        "content": "You are a read-only codebase search specialist.\n\nCRITICAL CONSTRAINTS:\n1. STRICTLY READ-ONLY: Use tools ONLY for reading. No modifications.\n2. EFFICIENCY: Spawn parallel tool calls for faster analysis.\n3. OUTPUT RULES: Use absolute paths. NO EMOJIS. No colons before tool calls.\n\nENV: Working directory is /Users/developer/workspace/code-analyzer (macOS, zsh)."
    },
    {
        "role": "user",
        "content": "TASK: Analyze aiohttp at `/Users/developer/workspace/code-analyzer/projects/aiohttp`. Explain its relationship with asyncio, key differences, and how they work together."
    },
    {
        "role": "assistant",
        "content": "I will start by listing the files and reading the documentation to understand the project structure.\n<tool_call>\n{\"name\": \"Bash\", \"arguments\": {\"command\": \"ls -F /Users/developer/workspace/code-analyzer/projects/aiohttp\", \"description\": \"List top-level directory\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"Read\", \"arguments\": {\"file_path\": \"/Users/developer/workspace/code-analyzer/projects/aiohttp/README.rst\"}}\n</tool_call>"
    },
    {
        "role": "user",
        "content": "<tool_response>\nFiles: CHANGES.rst, LICENSE.txt, README.rst, aiohttp/, docs/, examples/, pyproject.toml, setup.py\n\nREADME content excerpt: aiohttp is an asynchronous HTTP client/server framework built on top of asyncio. It supports both client and server side, Web-Sockets, and provides a web-server with middleware.\n</tool_response>\n\nBased on this, what are your next steps to analyze the deep integration with asyncio?"
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