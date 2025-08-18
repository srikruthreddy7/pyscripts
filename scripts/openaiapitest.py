#!/usr/bin/env python3
"""
Stream OpenAI Responses API and print reasoning summary while the model reasons.

Usage:
  python scripts/openaiapitest.py [question...]

Env:
  OPENAI_API_KEY must be set.
  OPENAI_MODEL (optional, defaults to 'gpt-5'; falls back to OPENAI_GPT5_MODEL if set)
  OPENAI_PRINT_REASONING_FULL=true to also stream full reasoning_text deltas (optional)
  NO_COLOR (any value) disables colored output
"""

import os
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False

try:
    from openai import OpenAI
except Exception:
    print("[error] openai package not installed. Run: pip install 'openai>=1.84.0'", file=sys.stderr)
    raise


def _supports_color(stream: Any) -> bool:
    """Return True if the given stream supports ANSI colors and NO_COLOR is not set."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    try:
        return bool(getattr(stream, "isatty", lambda: False)())
    except Exception:
        return False


class Ansi:
    reset = "\x1b[0m"
    bold = "\x1b[1m"
    dim = "\x1b[2m"
    green = "\x1b[32m"
    magenta = "\x1b[35m"
    cyan = "\x1b[36m"


def load_env_files() -> None:
    """Load .env and repo-root .env.local if present."""
    try:
        load_dotenv()
    except Exception:
        pass
    try:
        repo_root = Path(__file__).resolve().parents[1]
        local_env = repo_root / ".env.local"
        if local_env.exists():
            load_dotenv(dotenv_path=str(local_env))
    except Exception:
        pass


def _get_first_reasoning_summary_text(response: Any) -> str | None:
    """Return the first reasoning summary text if present."""
    items = getattr(response, "output", None) or []
    for item in items:
        item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
        if item_type == "reasoning":
            summary = getattr(item, "summary", None) or (item.get("summary") if isinstance(item, dict) else None) or []
            for s in summary:
                if hasattr(s, "text"):
                    if s.text:
                        return s.text
                elif isinstance(s, dict) and s.get("text"):
                    return s["text"]
    return None


def _get_final_answer_text(response: Any) -> str | None:
    """Collect the assistant's output_text content."""
    items = getattr(response, "output", None) or []
    parts: list[str] = []
    for item in items:
        item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
        if item_type == "message":
            content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else [])
            for c in content:
                c_type = getattr(c, "type", None) or (c.get("type") if isinstance(c, dict) else None)
                if c_type == "output_text":
                    text = getattr(c, "text", None) if hasattr(c, "text") else (c.get("text") if isinstance(c, dict) else None)
                    if text:
                        parts.append(text)
    return (" ".join(parts).strip()) if parts else None


def stream_and_print(client: OpenAI, model: str, question: str) -> Any:
    """Stream the response, printing reasoning summary and final answer as they arrive.

    - Prints reasoning summary deltas to stderr using response.reasoning_summary_* events
    - Prints final answer output_text to stdout
    - Returns the final Response object
    """
    show_full_reasoning = (os.environ.get("OPENAI_PRINT_REASONING_FULL", "").lower() in {"1", "true", "yes"})
    printed_reasoning_header = False

    # Prepare color support flags
    color_stderr = _supports_color(sys.stderr)
    color_stdout = _supports_color(sys.stdout)
    rs_prefix_printed = False

    with client.responses.stream(
        model=model,
        input=question,
        reasoning={"effort": "medium", "summary": "auto"},
        text={"verbosity": "medium"},
    ) as stream:
        for event in stream:
            etype = getattr(event, "type", None) or (event.get("type") if isinstance(event, dict) else None)

            # Reasoning summary lifecycle
            if etype == "response.reasoning_summary_part.added":
                if not printed_reasoning_header:
                    if color_stderr:
                        print(f"{Ansi.magenta}{Ansi.bold}Reasoning summary (live):{Ansi.reset}", file=sys.stderr, flush=True)
                    else:
                        print("Reasoning summary (live):", file=sys.stderr, flush=True)
                    printed_reasoning_header = True
            elif etype == "response.reasoning_summary_text.delta":
                delta = getattr(event, "delta", None) if hasattr(event, "delta") else (event.get("delta") if isinstance(event, dict) else None)
                if delta:
                    if color_stderr:
                        # Stream delta in magenta
                        print(f"{Ansi.magenta}{delta}{Ansi.reset}", end="", file=sys.stderr, flush=True)
                    else:
                        print(delta, end="", file=sys.stderr, flush=True)
            elif etype == "response.reasoning_summary_text.done":
                # Close the current summary part with a newline
                print(file=sys.stderr, flush=True)

            # Optional: full (non-summary) reasoning text
            elif etype == "response.reasoning_text.delta" and show_full_reasoning:
                delta = getattr(event, "delta", None) if hasattr(event, "delta") else (event.get("delta") if isinstance(event, dict) else None)
                if delta:
                    if color_stderr:
                        # Stream full reasoning in dim cyan
                        print(f"{Ansi.cyan}{Ansi.dim}{delta}{Ansi.reset}", end="", file=sys.stderr, flush=True)
                    else:
                        print(delta, end="", file=sys.stderr, flush=True)
            elif etype == "response.reasoning_text.done" and show_full_reasoning:
                print(file=sys.stderr, flush=True)

            # Assistant visible output text
            elif etype == "response.output_text.delta":
                delta = getattr(event, "delta", None) if hasattr(event, "delta") else (event.get("delta") if isinstance(event, dict) else None)
                if delta:
                    if color_stdout:
                        print(f"{Ansi.green}{delta}{Ansi.reset}", end="", flush=True)
                    else:
                        print(delta, end="", flush=True)
            elif etype == "response.output_text.done":
                print(flush=True)

            # Errors (if any)
            elif etype == "response.error":
                err = getattr(event, "error", None) if hasattr(event, "error") else (event.get("error") if isinstance(event, dict) else None)
                print(f"[error] {err}", file=sys.stderr, flush=True)

            # Other events are ignored

        # Retrieve the final response object for programmatic access if desired
        final_response = stream.get_final_response()
        return final_response


def main() -> None:
    load_env_files()
    if not os.environ.get("OPENAI_API_KEY"):
        print("[error] OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(2)

    model = os.environ.get("OPENAI_MODEL") or os.environ.get("OPENAI_GPT5_MODEL") or "gpt-5"
    question = " ".join(sys.argv[1:]).strip() or "What is the capital of France?"

    client = OpenAI()
    # Stream and print live reasoning summary and answer
    stream_and_print(client, model, question)


if __name__ == "__main__":
    main()


