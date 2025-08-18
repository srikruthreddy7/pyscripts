## OpenAI Responses Streaming Demo

Stream a response from the OpenAI Responses API while printing the model's reasoning summary live.

### Requirements
- Python 3.10+

### Setup
```bash
cd path/to/pyscripts
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
python -m pip install -U pip
python -m pip install "openai>=1.84.0" "python-dotenv>=1.0.1"
```

### Configure environment
Create `.env` in the repo root:
```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5
# Optional fallbacks/toggles
# OPENAI_GPT5_MODEL=gpt-5
# OPENAI_PRINT_REASONING_FULL=true   # stream full reasoning text
# NO_COLOR=1                         # disable ANSI colors
```
Notes:
- `.env.local` at the repo root is also loaded if present.

### Run
```bash
python scripts/openaiapitest.py "Say hi in one short sentence"
```
If no prompt is provided, a default question is used.

### Output behavior
- Reasoning summary is streamed to stderr.
- Final answer text is streamed to stdout.
- ANSI colors are used when supported; set `NO_COLOR` to disable.

Examples:
```bash
# Only show the final answer
python scripts/openaiapitest.py "Quick greeting" 2>/dev/null

# Only show the reasoning summary
python scripts/openaiapitest.py "Quick greeting" >/dev/null
```

### Troubleshooting
- "[error] OPENAI_API_KEY not set." → Ensure `.env` has `OPENAI_API_KEY` and the venv is activated.
- "[error] openai package not installed" → Re-run the install commands above.


