#!/usr/bin/env python3
"""
debug_parallel_judge_hvac_gpt5.py

DEBUG VERSION: Shows exactly what's happening with each row and OpenAI request.
This version adds extensive logging to help debug issues.
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
# DEBUG-ENHANCED Async judge worker
# ---------------------------

async def judge_one_debug(
    client: AsyncOpenAI,
    model: str,
    transcript: str,
    row_index: int,
    max_output_tokens: int,
    max_attempts: int = 3,
    max_prompt_tokens: int = 0,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int], Optional[str]]:
    """DEBUG VERSION: Shows all processing steps"""
    
    print(f"\n{'='*80}")
    print(f"🔍 PROCESSING ROW {row_index}")
    print(f"{'='*80}")
    
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    if not transcript or not transcript.strip():
        print(f"❌ ROW {row_index}: EMPTY TRANSCRIPT")
        return None, usage, "empty_transcript"

    print(f"📝 ORIGINAL TRANSCRIPT (Row {row_index}):")
    print(f"Length: {len(transcript)} characters")
    print("-" * 40)
    print(transcript[:500] + ("..." if len(transcript) > 500 else ""))
    print("-" * 40)

    if max_prompt_tokens > 0:
        original_len = len(transcript)
        transcript = compact_transcript_if_needed(transcript, model, max_prompt_tokens)
        if len(transcript) != original_len:
            print(f"✂️ TRANSCRIPT COMPACTED: {original_len} -> {len(transcript)} chars")

    # Create messages
    msgs = make_messages(transcript)
    response_format = make_response_format()
    
    print(f"\n📤 MESSAGES TO OPENAI (Row {row_index}):")
    for i, msg in enumerate(msgs):
        print(f"Message {i+1} ({msg['role']}):")
        print("-" * 30)
        print(msg['content'][:300] + ("..." if len(msg['content']) > 300 else ""))
        print("-" * 30)
    
    print(f"\n🔧 REQUEST PARAMETERS:")
    print(f"Model: {model}")
    print(f"Max output tokens: {max_output_tokens}")
    print(f"Temperature: 0.0")
    print(f"Response format: JSON Schema (strict)")
    
    print(f"\n📋 DETAILED REQUEST STRUCTURE:")
    print(f"API Endpoint: client.responses.create()")
    print(f"Full parameter dict:")
    request_params = {
        "model": model,
        "reasoning": {"effort": "high"},
        "input": msgs,
        "response_format": response_format,
        "max_output_tokens": max_output_tokens,
        "temperature": 0.0,
    }
    for key, value in request_params.items():
        print(f"  {key}: {type(value)} = {str(value)[:200]}{'...' if len(str(value)) > 200 else ''}")
    
    print(f"\n🔍 RESPONSE_FORMAT DETAILS:")
    print(f"Type: {type(response_format)}")
    print(f"Content: {json.dumps(response_format, indent=2)}")
    
    print(f"\n🔍 REASONING PARAMETER DETAILS:")
    reasoning_param = {"effort": "high"}
    print(f"Type: {type(reasoning_param)}")
    print(f"Content: {reasoning_param}")

    for attempt in range(1, max_attempts + 1):
        print(f"\n🚀 ATTEMPT {attempt} (Row {row_index})")
        try:
            print(f"Calling: client.responses.create() with parameters:")
            print(f"  model='{model}'")
            print(f"  reasoning={reasoning_param}")
            print(f"  input={len(msgs)} messages")
            print(f"  response_format={type(response_format)} object")
            print(f"  max_output_tokens={max_output_tokens}")
            print(f"  temperature=0.0")
            print(f"Executing API call now...")
            
            resp = await client.responses.create(
                model=model,
                reasoning={"effort": "high"},
                input=msgs,
                response_format=response_format,
                max_output_tokens=max_output_tokens,
                temperature=0.0,
            )
            
            print(f"✅ Received response from OpenAI")
            
            # Log usage
            u = getattr(resp, "usage", None)
            if u:
                input_tokens = getattr(u, "input_tokens", 0) or 0
                output_tokens = getattr(u, "output_tokens", 0) or 0
                total_tokens = getattr(u, "total_tokens", 0) or 0
                
                usage["input_tokens"] += input_tokens
                usage["output_tokens"] += output_tokens
                usage["total_tokens"] += total_tokens
                
                print(f"📊 TOKEN USAGE: Input={input_tokens}, Output={output_tokens}, Total={total_tokens}")

            # Debug the response object
            print(f"\n📥 RAW RESPONSE OBJECT:")
            print(f"Type: {type(resp)}")
            print(f"Has output_parsed: {hasattr(resp, 'output_parsed')}")
            print(f"Has output_text: {hasattr(resp, 'output_text')}")
            print(f"Has output: {hasattr(resp, 'output')}")
            
            # Try to extract JSON
            try:
                parsed = extract_json_from_response(resp)
                print(f"✅ SUCCESSFULLY EXTRACTED JSON:")
                print(json.dumps(parsed, indent=2)[:500] + ("..." if len(str(parsed)) > 500 else ""))
                
                # Validate against schema
                try:
                    obj = HVACJudgeResult.model_validate(parsed).model_dump(mode="json")
                    print(f"✅ JSON VALIDATION SUCCESSFUL")
                    print(f"Overall score: {obj.get('overall', 'N/A')}")
                    return obj, usage, None
                    
                except ValidationError as ve:
                    print(f"❌ JSON VALIDATION FAILED:")
                    print(f"Error: {ve}")
                    print(f"Attempting repair...")
                    
                    repair_msgs = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Your previous JSON failed validation:\n{ve}\n\nReturn corrected JSON ONLY, strictly following the schema."}
                    ]
                    
                    print(f"🔧 SENDING REPAIR REQUEST...")
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
                    print(f"✅ REPAIR RESPONSE:")
                    print(json.dumps(parsed2, indent=2)[:500] + ("..." if len(str(parsed2)) > 500 else ""))
                    
                    obj2 = HVACJudgeResult.model_validate(parsed2).model_dump(mode="json")
                    print(f"✅ REPAIR VALIDATION SUCCESSFUL")
                    return obj2, usage, None
                    
            except Exception as json_err:
                print(f"❌ JSON EXTRACTION FAILED:")
                print(f"Error: {json_err}")
                print(f"Response details:")
                if hasattr(resp, 'output_text'):
                    print(f"output_text: {getattr(resp, 'output_text', 'None')}")
                if hasattr(resp, 'output_parsed'):
                    print(f"output_parsed: {getattr(resp, 'output_parsed', 'None')}")
                raise json_err

        except (APIError, RateLimitError, APITimeoutError, InternalServerError) as e:
            print(f"❌ API ERROR (attempt {attempt}):")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            await asyncio.sleep(min(2 ** attempt, 20))
            last_err = f"{type(e).__name__}: {e}"
            
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR (attempt {attempt}):")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            await asyncio.sleep(min(2 ** attempt, 20))
            last_err = f"{type(e).__name__}: {e}"

    print(f"❌ ALL ATTEMPTS FAILED for row {row_index}")
    return None, usage, last_err


# ---------------------------
# DEBUG Orchestrator
# ---------------------------

async def run_async_debug(args) -> None:
    print(f"🚀 STARTING DEBUG RUN")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Model: {args.model}")
    print(f"Min chars: {args.min_chars}")
    print(f"Max concurrency: {args.max_concurrency}")
    
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print(f"✅ OpenAI client initialized")

    print(f"\n📁 LOADING CSV...")
    df = pd.read_csv(args.input)
    print(f"✅ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    tcol = find_transcript_column(df)
    print(f"✅ Found transcript column: '{tcol}'")

    ensure_dir(args.output)

    OUT_COL = args.eval_column
    if OUT_COL not in df.columns:
        df[OUT_COL] = ""

    # Find candidates
    candidates: List[int] = []
    print(f"\n🔍 FINDING CANDIDATE ROWS...")
    for idx, val in df[tcol].items():
        print(f"Row {idx}: ", end="")
        if not isinstance(val, str):
            print(f"❌ Not a string (type: {type(val)})")
            continue
        if len(val) < args.min_chars:
            print(f"❌ Too short ({len(val)} < {args.min_chars} chars)")
            continue
        if args.resume:
            existing = df.at[idx, OUT_COL]
            if isinstance(existing, str) and existing.strip():
                print(f"⏭️ Already processed")
                continue
        print(f"✅ Valid ({len(val)} chars)")
        candidates.append(idx)

    total_candidates = len(candidates)
    print(f"\n📋 SUMMARY:")
    print(f"Total rows: {len(df)}")
    print(f"Candidate rows: {total_candidates}")
    print(f"Candidates: {candidates}")

    if total_candidates == 0:
        print("❌ No rows to process!")
        return

    # For debugging, let's just process the FIRST row only
    print(f"\n🔄 PROCESSING ONLY FIRST ROW FOR DEBUGGING...")
    
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    # Only process the first candidate row
    row_idx = candidates[0]
    print(f"\n🎯 PROCESSING ONLY ROW {row_idx}")
    
    transcript = df.at[row_idx, tcol]
    print(f"Transcript preview: {transcript[:100]}...")
    
    result_json, usage, err = await judge_one_debug(
        client=client,
        model=args.model,
        transcript=transcript,
        row_index=row_idx,
        max_output_tokens=args.max_output_tokens,
        max_attempts=args.max_attempts,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    
    if result_json is not None:
        df.at[row_idx, OUT_COL] = json.dumps(result_json, ensure_ascii=False)
        print(f"✅ Row {row_idx} completed successfully")
    else:
        df.at[row_idx, OUT_COL] = ""
        print(f"❌ Row {row_idx} failed: {err}")

    # Update total usage
    for k in total_usage:
        total_usage[k] += usage.get(k, 0)
    
    print(f"💾 Saving result...")
    df.to_csv(args.output, index=False)

    print(f"\n✅ ALL ROWS PROCESSED")
    df.to_csv(args.output, index=False)
    print(f"📁 Final output saved to: {args.output}")

    # Print usage summary
    IN_PER_M = 1.25
    OUT_PER_M = 10.00
    in_cost = total_usage["input_tokens"]/1_000_000 * IN_PER_M
    out_cost = total_usage["output_tokens"]/1_000_000 * OUT_PER_M
    
    print(f"\n📊 FINAL USAGE SUMMARY:")
    print(f"Input tokens: {total_usage['input_tokens']:,}")
    print(f"Output tokens: {total_usage['output_tokens']:,}")
    print(f"Total tokens: {total_usage['total_tokens']:,}")
    print(f"Estimated cost: ${(in_cost+out_cost):.4f}")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DEBUG: Parallel LLM-as-a-Judge for HVAC transcripts using GPT-5 (high reasoning).")
    p.add_argument("--input", required=True, help="Path to input CSV with a 'Transcript' column")
    p.add_argument("--output", required=True, help="Path to write the CSV containing the 'gpt-5-high-eval' column")
    p.add_argument("--model", default="gpt-5", help="Judge model (default: gpt-5)")
    p.add_argument("--min-chars", type=int, default=300, help="Skip transcripts shorter than this many characters")
    p.add_argument("--max-concurrency", type=int, default=12, help="Max concurrent API requests (ignored in debug mode)")
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
        asyncio.run(run_async_debug(args))
    except KeyboardInterrupt:
        print("\n[warn] Interrupted by user. Partial results (if any) are in:", args.output)

if __name__ == "__main__":
    main()
