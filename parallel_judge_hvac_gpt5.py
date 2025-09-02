#!/usr/bin/env python3
"""
parallel_judge_hvac_gpt5.py

Asynchronous "LLM-as-a-Judge" pipeline for HVAC call transcripts using
OpenAI Responses API with GPT-5 (reasoning.effort="high") + Strict Structured Outputs.

- Filters transcripts by minimum character length (default: 300).
- Runs evaluations in parallel (async) with a concurrency limit.
- Validates model output against a JSON Schema (Pydantic).
- Writes a new column "gpt-5-high-eval" (raw JSON per row) to the output CSV.
- Optionally writes a companion *.scored.csv with numeric convenience columns.
- Prints token usage totals (and an estimated cost) at the end.

Usage:
  export OPENAI_API_KEY=sk-...

  pip install --upgrade openai pandas pydantic tqdm numpy tiktoken

  python parallel_judge_hvac_gpt5.py \
    --input /path/calls.csv \
    --output /path/calls_with_judgments.csv \
    --model gpt-5 \
    --min-chars 300 \
    --max-concurrency 16 \
    --max-output-tokens 800 \
    --resume \
    --checkpoint-every 100 \
    --write-scored

Notes:
- Requires OpenAI Python SDK v1.50+.
- If your transcripts are extremely long, set --max-prompt-tokens (requires tiktoken).
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator
from tqdm import tqdm

# Optional tokenization (for prompt trimming)
try:
    import tiktoken
except Exception:
    tiktoken = None

# OpenAI async client
try:
    from openai import AsyncOpenAI
    from openai import APIError, RateLimitError, APITimeoutError, InternalServerError
except Exception as e:
    print("ERROR: Install the OpenAI SDK >= 1.50.0  ->  pip install --upgrade openai", file=sys.stderr)
    raise


# ---------------------------
# Schema for strict JSON outputs
# ---------------------------

class ScoreBlock(BaseModel):
    diagnostic_rigor: int = Field(..., ge=1, le=7)
    technical_accuracy: int = Field(..., ge=1, le=7)
    procedural_safety: int = Field(..., ge=1, le=7)
    actionability: int = Field(..., ge=1, le=7)
    communication: int = Field(..., ge=1, le=7)
    evidence_use: int = Field(..., ge=1, le=7)
    resolution_quality: int = Field(..., ge=1, le=7)

class FlagsBlock(BaseModel):
    unsafe_instruction: bool
    hallucination_signals: bool
    pii_present: bool
    missing_outcome: bool
    format_issue: bool

class OutcomeBlock(BaseModel):
    status: str = Field(..., pattern=r"^(resolved|partially_resolved|unresolved|triaged)$")
    summary: str = Field(..., min_length=0, max_length=800)

class KeySignals(BaseModel):
    measurements_present: bool
    codes_or_readings: bool
    parts_or_tools_specified: bool
    model_or_unit_info_used: bool

class HVACJudgeResult(BaseModel):
    scores: ScoreBlock
    flags: FlagsBlock
    outcome: OutcomeBlock
    key_signals: KeySignals
    clips: list[str] = Field(default_factory=list, max_length=3)  # <= 3 short supporting quotes
    overall: int = Field(..., ge=1, le=7)
    justification: str = Field(..., min_length=1, max_length=400)

    @field_validator("clips")
    @classmethod
    def _limit_clips(cls, v):
        return v[:3]

def strict_schema() -> Dict[str, Any]:
    return HVACJudgeResult.model_json_schema()

def compute_weighted_score(s: ScoreBlock) -> float:
    return round(
        0.25*s.technical_accuracy +
        0.20*s.diagnostic_rigor +
        0.15*s.procedural_safety +
        0.15*s.actionability +
        0.10*s.evidence_use +
        0.10*s.communication +
        0.05*s.resolution_quality,
        3
    )


# ---------------------------
# Prompts
# ---------------------------

SYSTEM_PROMPT = """\
You are an impartial HVAC QA judge. Evaluate how well the Expert AI assists a field HVAC technician in a real call transcript.
Be strict and reward accurate, safe, actionable, measurement-driven guidance. Penalize vagueness, unsafe steps, or speculation.

Important:
- Output ONLY JSON matching the provided schema exactly.
- Do NOT reveal chain-of-thought. Provide a short, evidence-based justification.
- If the transcript lacks details, score conservatively and set flags.
"""

USER_PROMPT_TEMPLATE = """\
[TRANSCRIPT]
{transcript}

[RUBRIC — score 1–7 (higher is better)]
- technical_accuracy: HVAC facts and typical ranges; no hallucinated specs/parts.
- diagnostic_rigor: Structured triage (airflow vs charge vs electrical vs controls).
- procedural_safety: Lockout/tagout, voltage/pressure cautions, refrigerant handling.
- actionability: Concrete steps, tools, readings, decision branches; clear next steps.
- communication: Clarity, structure, brevity.
- evidence_use: Uses measurements, error codes, nameplate data, observed symptoms.
- resolution_quality: Resolved or credibly triaged with realistic plan.

[FLAGS]
- unsafe_instruction, hallucination_signals, pii_present, missing_outcome, format_issue

[KEY SIGNALS]
- measurements_present, codes_or_readings, parts_or_tools_specified, model_or_unit_info_used

[OUTPUT]
Return valid JSON that strictly conforms to the schema. Include up to 3 short supporting quotes in "clips" (≤160 chars each).
"""


# ---------------------------
# Helpers
# ---------------------------

def find_transcript_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.strip().lower() == "transcript":
            return c
    raise KeyError("Could not find a 'Transcript' column (case-insensitive).")

def make_messages(transcript: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(transcript=transcript)},
    ]

def make_response_format() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "HVACJudgeResult",
            "strict": True,
            "schema": strict_schema(),
        },
    }

def extract_json_from_response(resp: Any) -> Dict[str, Any]:
    """
    Robustly extract the structured JSON from a Responses API response.
    Tries:
      - resp.output_parsed (preferred for structured outputs)
      - resp.output_text (single aggregated string)
      - walk resp.output -> message -> content -> text/value
    """
    parsed = getattr(resp, "output_parsed", None)
    if parsed:
        return parsed
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return json.loads(txt)
    out = getattr(resp, "output", None) or []
    chunks: List[str] = []
    for item in out:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", None) or getattr(c, "value", None)
                if t:
                    chunks.append(t)
    if chunks:
        return json.loads("".join(chunks))
    raise ValueError("No JSON content found in response object.")

def ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _count_tokens(text: str, model: str) -> int:
    if not tiktoken:
        return max(1, int(len(text) / 4))
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def compact_transcript_if_needed(transcript: str, model: str, max_prompt_tokens: int) -> str:
    if max_prompt_tokens <= 0 or not tiktoken:
        return transcript
    tok = _count_tokens(transcript, model)
    if tok <= max_prompt_tokens:
        return transcript
    enc = tiktoken.encoding_for_model(model) if tiktoken else None
    if not enc:
        return transcript[: max_prompt_tokens * 4]
    toks = enc.encode(transcript)
    half = max_prompt_tokens // 2
    head = enc.decode(toks[:half])
    tail = enc.decode(toks[-half:])
    return head + "\n...\n" + tail


# ---------------------------
# Async judge worker
# ---------------------------

async def judge_one(
    client: AsyncOpenAI,
    model: str,
    transcript: str,
    max_output_tokens: int,
    max_attempts: int = 3,
    max_prompt_tokens: int = 0,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int], Optional[str]]:
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if not transcript or not transcript.strip():
        return None, usage, "empty_transcript"

    if max_prompt_tokens > 0:
        transcript = compact_transcript_if_needed(transcript, model, max_prompt_tokens)

    msgs = make_messages(transcript)
    response_format = make_response_format()

    for attempt in range(1, max_attempts + 1):
        try:
            resp = await client.responses.create(
                model=model,
                reasoning={"effort": "high"},
                input=msgs,
                response_format=response_format,
                max_output_tokens=max_output_tokens,
                temperature=0.0,
            )
            u = getattr(resp, "usage", None)
            if u:
                usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
                usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
                usage["total_tokens"] += getattr(u, "total_tokens", 0) or 0

            parsed = extract_json_from_response(resp)
            try:
                obj = HVACJudgeResult.model_validate(parsed).model_dump(mode="json")
                return obj, usage, None
            except ValidationError as ve:
                repair_msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Your previous JSON failed validation:\n{ve}\n\nReturn corrected JSON ONLY, strictly following the schema."}
                ]
                resp2 = await client.responses.create(
                    model=model,
                    reasoning={"effort": "high"},
                    input=repair_msgs,
                    response_format=response_format,
                    max_output_tokens=max_output_tokens,
                    temperature=0.0,
                )
                u2 = getattr(resp2, "usage", None)
                if u2:
                    usage["input_tokens"] += getattr(u2, "input_tokens", 0) or 0
                    usage["output_tokens"] += getattr(u2, "output_tokens", 0) or 0
                    usage["total_tokens"] += getattr(u2, "total_tokens", 0) or 0
                parsed2 = extract_json_from_response(resp2)
                obj2 = HVACJudgeResult.model_validate(parsed2).model_dump(mode="json")
                return obj2, usage, None

        except (APIError, RateLimitError, APITimeoutError, InternalServerError) as e:
            await asyncio.sleep(min(2 ** attempt, 20))
            last_err = f"{type(e).__name__}: {e}"
        except Exception as e:
            await asyncio.sleep(min(2 ** attempt, 20))
            last_err = f"{type(e).__name__}: {e}"

    return None, usage, last_err


# ---------------------------
# Orchestrator
# ---------------------------

async def run_async(args) -> None:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    df = pd.read_csv(args.input)
    tcol = find_transcript_column(df)

    ensure_dir(args.output)

    OUT_COL = args.eval_column
    if OUT_COL not in df.columns:
        df[OUT_COL] = ""

    candidates: List[int] = []
    for idx, val in df[tcol].items():
        if not isinstance(val, str):
            continue
        if len(val) < args.min_chars:
            continue
        if args.resume:
            existing = df.at[idx, OUT_COL]
            if isinstance(existing, str) and existing.strip():
                continue
        candidates.append(idx)

    total_candidates = len(candidates)
    print(f"[info] Rows to evaluate: {total_candidates} (min_chars={args.min_chars})")

    sem = asyncio.Semaphore(args.max_concurrency)
    pbar = tqdm(total=total_candidates, desc="Evaluating", ncols=100)

    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    usage_lock = asyncio.Lock()

    async def worker(row_idx: int):
        transcript = df.at[row_idx, tcol]
        async with sem:
            result_json, usage, err = await judge_one(
                client=client,
                model=args.model,
                transcript=transcript,
                max_output_tokens=args.max_output_tokens,
                max_attempts=args.max_attempts,
                max_prompt_tokens=args.max_prompt_tokens,
            )
        if result_json is not None:
            df.at[row_idx, OUT_COL] = json.dumps(result_json, ensure_ascii=False)
        else:
            df.at[row_idx, OUT_COL] = ""

        async with usage_lock:
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)

        if args.checkpoint_every and (pbar.n + 1) % args.checkpoint_every == 0:
            df.to_csv(args.output, index=False)

        pbar.update(1)

    tasks = [asyncio.create_task(worker(i)) for i in candidates]
    await asyncio.gather(*tasks)
    pbar.close()

    df.to_csv(args.output, index=False)
    print(f"[done] Wrote: {args.output}")

    if args.write_scored:
        rows = []
        for idx in candidates:
            raw = df.at[idx, OUT_COL]
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                data = json.loads(raw)
                parsed = HVACJudgeResult.model_validate(data)
                s = parsed.scores
                rows.append({
                    "row_index": idx,
                    "weighted_score": compute_weighted_score(s),
                    "overall": parsed.overall,
                    "technical_accuracy": s.technical_accuracy,
                    "diagnostic_rigor": s.diagnostic_rigor,
                    "procedural_safety": s.procedural_safety,
                    "actionability": s.actionability,
                    "communication": s.communication,
                    "evidence_use": s.evidence_use,
                    "resolution_quality": s.resolution_quality,
                    "flags": parsed.flags.model_dump(),
                    "outcome_status": parsed.outcome.status,
                    "key_signals": parsed.key_signals.model_dump(),
                })
            except Exception:
                pass
        if rows:
            scored_df = pd.DataFrame(rows).sort_values(["weighted_score", "overall", "technical_accuracy"], ascending=False)
            scored_path = os.path.splitext(args.output)[0] + ".scored.csv"
            scored_df.to_csv(scored_path, index=False)
            print(f"[info] Wrote scored file: {scored_path}")

    IN_PER_M = 1.25
    OUT_PER_M = 10.00
    in_cost = total_usage["input_tokens"]/1_000_000 * IN_PER_M
    out_cost = total_usage["output_tokens"]/1_000_000 * OUT_PER_M
    print(f"[usage] input={total_usage['input_tokens']:,}  output={total_usage['output_tokens']:,}  total={total_usage['total_tokens']:,}")
    print(f"[est. cost] input=${in_cost:.4f}  output=${out_cost:.4f}  total=${(in_cost+out_cost):.4f}")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parallel LLM-as-a-Judge for HVAC transcripts using GPT-5 (high reasoning).")
    p.add_argument("--input", required=True, help="Path to input CSV with a 'Transcript' column")
    p.add_argument("--output", required=True, help="Path to write the CSV containing the 'gpt-5-high-eval' column")
    p.add_argument("--model", default="gpt-5", help="Judge model (default: gpt-5)")
    p.add_argument("--min-chars", type=int, default=300, help="Skip transcripts shorter than this many characters")
    p.add_argument("--max-concurrency", type=int, default=12, help="Max concurrent API requests")
    p.add_argument("--max-attempts", type=int, default=3, help="Max attempts per row (with backoff)")
    p.add_argument("--max-output-tokens", type=int, default=800, help="Max output tokens per request")
    p.add_argument("--max-prompt-tokens", type=int, default=0, help="If >0 (and tiktoken installed), compact transcripts above this token length")
    p.add_argument("--resume", action="store_true", help="Skip rows that already have JSON in the output column")
    p.add_argument("--checkpoint-every", type=int, default=100, help="Write partial CSV every N completed rows")
    p.add_argument("--eval-column", default="gpt-5-high-eval", help="Name of the output JSON column")
    p.add_argument("--write-scored", action="store_true", help="Also write *.scored.csv with numeric convenience columns")
    return p

def main():
    args = build_argparser().parse_args()
    try:
        import asyncio
        asyncio.run(run_async(args))
    except KeyboardInterrupt:
        print("\n[warn] Interrupted by user. Partial results (if any) are in:", args.output)

if __name__ == "__main__":
    main()
