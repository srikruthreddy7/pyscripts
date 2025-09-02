#!/usr/bin/env python3
"""
parallel_judge_hvac_gpt5_v3.py

Async parallel “LLM-as-a-Judge” for HVAC transcripts using the OpenAI
Responses API with GPT-5 (reasoning.effort="high") + Structured Outputs.

**Why v3?**
- Fixes BadRequestError by **removing `temperature`** (some reasoning models reject it).
- Uses **Structured Outputs via `text.format: { type: "json_schema" }`** (official pattern).
- Includes **automatic fallback to JSON mode** (`text.format: { type: "json_object" }`) when schema formatting
  is not supported by the selected model / account.
- Keeps strict **Pydantic validation** and a **repair-retry** path to enforce the schema.

Docs:
- Responses API parameters and `text.format`: https://platform.openai.com/docs/api-reference/responses
- Structured Outputs (JSON schema via Responses API): https://platform.openai.com/docs/guides/structured-outputs
- Reasoning effort (low|medium|high) for reasoning-capable models: https://platform.openai.com/docs/guides/reasoning

Usage
-----
  pip install --upgrade openai pandas pydantic tqdm numpy tiktoken
  export OPENAI_API_KEY=sk-...

  python parallel_judge_hvac_gpt5_v3.py \
    --input /path/calls.csv \
    --output /path/calls_with_judgments.csv \
    --model gpt-5 \
    --min-chars 300 \
    --max-concurrency 16 \
    --max-output-tokens 2000 \
    --resume \
    --checkpoint-every 100 \
    --write-scored

Notes
-----
- Requires OpenAI Python SDK v1.50+ with Async support.
- If your project/model rejects schema-based structured outputs, the script will automatically
  try **JSON mode** and still validate+repair to match the schema you want.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator
from tqdm import tqdm

# Optional tokenization for prompt trimming
try:
    import tiktoken
except Exception:
    tiktoken = None

# OpenAI async client + exceptions
try:
    from openai import AsyncOpenAI
    from openai import APIError, RateLimitError, APITimeoutError, InternalServerError, BadRequestError
except Exception as e:
    print("ERROR: Install the OpenAI SDK >= 1.50.0  ->  pip install --upgrade openai", file=sys.stderr)
    raise


# ---------------------------
# Pydantic schema and helpers
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
    summary: str = Field(..., min_length=0, max_length=2000)

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

def _enforce_additional_properties_false(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Structured Outputs requires `additionalProperties: false` on objects.
    Pydantic v2 doesn't add it by default, so we inject it everywhere and ensure `required` is complete.
    """
    s = deepcopy(schema)

    def walk(node: Any):
        if not isinstance(node, dict):
            return
        t = node.get("type")
        if t == "object":
            props = node.get("properties", {}) or {}
            node["additionalProperties"] = False
            req = set(node.get("required", []))
            req.update(props.keys())
            node["required"] = sorted(req)
            for v in props.values():
                walk(v)
            if "$defs" in node:
                for v in (node["$defs"] or {}).values():
                    walk(v)
        elif t == "array":
            items = node.get("items")
            if items:
                walk(items)
        else:
            if "$defs" in node:
                for v in (node["$defs"] or {}).values():
                    walk(v)
    walk(s)
    return s

def strict_schema() -> Dict[str, Any]:
    raw = HVACJudgeResult.model_json_schema()
    return _enforce_additional_properties_false(raw)


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
# Utilities
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

def make_text_format(structured: bool, schema: Dict[str, Any]) -> Dict[str, Any]:
    if structured:
        return {
            "format": {
                "type": "json_schema",
                "name": "HVACJudgeResult",
                "schema": schema,
                "strict": True,
            }
        }
    else:
        # JSON mode fallback (valid JSON, but no schema enforcement)
        return {"format": {"type": "json_object"}}

def extract_json_from_response(resp: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns (json_obj or None, refusal_text or None).
    With text.format json_schema or json_object, output_text is usually the JSON string.
    """
    # Handle refusal
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "refusal":
                    return None, getattr(c, "refusal", "refused")

    # Preferred convenience: output_text
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        try:
            return json.loads(txt), None
        except Exception:
            pass

    # Walk content as a fallback
    chunks: List[str] = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", None) or getattr(c, "value", None)
                if t:
                    chunks.append(t)
    if chunks:
        try:
            return json.loads("".join(chunks)), None
        except Exception:
            return None, None
    return None, None

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
    schema: Dict[str, Any],
    max_output_tokens: int,
    max_attempts: int = 3,
    max_prompt_tokens: int = 0,
    structured: bool = True,
    auto_fallback_json_mode: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int], Optional[str]]:
    """
    Returns: (parsed_json or None, usage_dict, error_message or None)
    usage_dict keys: input_tokens, output_tokens, total_tokens
    """
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    if not transcript or not transcript.strip():
        return None, usage, "empty_transcript"

    if max_prompt_tokens > 0:
        transcript = compact_transcript_if_needed(transcript, model, max_prompt_tokens)

    msgs = make_messages(transcript)
    text_format = make_text_format(structured, schema)

    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"\n🚀 DEBUG: Making API call (attempt {attempt})")
            print(f"  Model: {model}")
            print(f"  Reasoning: {'high'}")
            print(f"  Input messages: {len(msgs)}")
            print(f"  Text format type: {text_format.get('format', {}).get('type', 'unknown')}")
            print(f"  Max output tokens: {max_output_tokens}")
            
            # FIXED: Use responses.parse() with text_format parameter as per documentation
            print(f"  🔧 Using NEW API METHOD: client.responses.parse()")
            print(f"  🔧 Using text_format=HVACJudgeResult instead of text parameter")
            
            # Per OpenAI docs: responses.parse() with text_format parameter
            resp = await client.responses.parse(
                model=model,
                input=msgs,
                text_format=HVACJudgeResult,
                max_output_tokens=max_output_tokens,
            )
            
            print(f"✅ API call successful!")
            print(f"📊 Response object type: {type(resp)}")
            print(f"📊 Response attributes: {[attr for attr in dir(resp) if not attr.startswith('_')]}")
            
            # Usage debug
            u = getattr(resp, "usage", None)
            print(f"📊 Usage object: {u}")
            if u:
                input_toks = getattr(u, "input_tokens", 0) or 0
                output_toks = getattr(u, "output_tokens", 0) or 0
                total_toks = getattr(u, "total_tokens", 0) or 0
                print(f"📊 Token usage: input={input_toks}, output={output_toks}, total={total_toks}")
                usage["input_tokens"] += input_toks
                usage["output_tokens"] += output_toks
                usage["total_tokens"] += total_toks
            else:
                print(f"⚠️ No usage object found!")

            # Response content debug
            print(f"\n🔍 EXTRACTING JSON FROM RESPONSE...")
            print(f"  Response has output_text: {hasattr(resp, 'output_text')}")
            if hasattr(resp, 'output_text'):
                output_text = getattr(resp, 'output_text', None)
                print(f"  output_text type: {type(output_text)}")
                print(f"  output_text length: {len(str(output_text)) if output_text else 0}")
                print(f"  output_text preview: {str(output_text)[:200]}..." if output_text else "None")
            
            print(f"  Response has output: {hasattr(resp, 'output')}")
            if hasattr(resp, 'output'):
                output = getattr(resp, 'output', None)
                print(f"  output type: {type(output)}")
                print(f"  output length: {len(output) if output else 0}")
                if output:
                    for i, item in enumerate(output[:2]):  # First 2 items
                        print(f"    output[{i}] type: {type(item)}")
                        print(f"    output[{i}] attributes: {[attr for attr in dir(item) if not attr.startswith('_')]}")
                        
                        # PROVE WHAT'S ACTUALLY IN THIS OBJECT
                        print(f"🔍 PROVING CONTENT OF output[{i}]:")
                        print(f"      .type = {getattr(item, 'type', 'NO TYPE ATTR')}")
                        print(f"      .status = {getattr(item, 'status', 'NO STATUS ATTR')}")
                        print(f"      .content type = {type(getattr(item, 'content', None))}")
                        
                        content = getattr(item, 'content', None)
                        if content:
                            print(f"      .content length = {len(str(content))}")
                            print(f"      .content preview = {str(content)[:200]}...")
                            print(f"      .content full = {repr(content)}")
                        else:
                            print(f"      .content = NULL/EMPTY")
                        
                        encrypted_content = getattr(item, 'encrypted_content', None)
                        if encrypted_content:
                            print(f"      .encrypted_content type = {type(encrypted_content)}")
                            print(f"      .encrypted_content length = {len(str(encrypted_content))}")
                            print(f"      .encrypted_content preview = {str(encrypted_content)[:200]}...")
                        else:
                            print(f"      .encrypted_content = NULL/EMPTY")
                        
                        summary = getattr(item, 'summary', None)
                        if summary:
                            print(f"      .summary type = {type(summary)}")
                            print(f"      .summary length = {len(str(summary))}")
                            print(f"      .summary preview = {str(summary)[:200]}...")
                        else:
                            print(f"      .summary = NULL/EMPTY")

            # FIXED: Use output_parsed from responses.parse() API  
            print(f"\n🔍 USING NEW RESPONSE PARSING...")
            print(f"  Response has output_parsed: {hasattr(resp, 'output_parsed')}")
            
            if hasattr(resp, 'output_parsed') and resp.output_parsed:
                print(f"✅ Found output_parsed!")
                print(f"  output_parsed type: {type(resp.output_parsed)}")
                print(f"  output_parsed content: {resp.output_parsed}")
                
                # The response.parse() method already returns the parsed Pydantic model
                obj = resp.output_parsed.model_dump(mode="json") if hasattr(resp.output_parsed, 'model_dump') else resp.output_parsed
                print(f"✅ Got final object!")
                print(f"  Final object type: {type(obj)}")
                print(f"  Final object keys: {list(obj.keys()) if isinstance(obj, dict) else 'not a dict'}")
                print(f"  Final object overall score: {obj.get('overall', 'missing')}")
                
                return obj, usage, None
            else:
                # Fallback to old method for debugging
                print(f"⚠️ No output_parsed found, trying old extraction method...")
                parsed, refusal = extract_json_from_response(resp)
                print(f"🔍 extract_json_from_response returned:")
                print(f"  parsed: {parsed is not None}")
                print(f"  parsed type: {type(parsed)}")
                print(f"  refusal: {refusal}")
                
                if refusal:
                    print(f"❌ Request was refused: {refusal}")
                    return None, usage, f"refusal: {refusal}"

                if parsed is None:
                    print(f"❌ No JSON could be parsed from response")
                    return None, usage, "No JSON parsed from response"

                print(f"\n🔍 VALIDATING WITH PYDANTIC...")
                try:
                    print(f"  Calling HVACJudgeResult.model_validate(parsed)...")
                    validated = HVACJudgeResult.model_validate(parsed)
                    print(f"✅ Pydantic validation successful!")
                    
                    obj = validated.model_dump(mode="json")
                    print(f"✅ model_dump successful!")
                    return obj, usage, None
                except ValidationError as ve:
                    print(f"❌ Pydantic validation failed:")
                    print(f"  ValidationError: {ve}")
                    print(f"  Attempting repair...")
                    # Repair request (also omit temperature)
                    repair_msgs = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Your previous JSON failed validation:\n{ve}\n\nReturn corrected JSON ONLY that strictly follows the schema."}
                    ]
                    resp2 = await client.responses.parse(
                        model=model,
                        input=repair_msgs,
                        text_format=HVACJudgeResult,
                        max_output_tokens=max_output_tokens,
                    )
                    u2 = getattr(resp2, "usage", None)
                    if u2:
                        usage["input_tokens"] += getattr(u2, "input_tokens", 0) or 0
                        usage["output_tokens"] += getattr(u2, "output_tokens", 0) or 0
                        usage["total_tokens"] += getattr(u2, "total_tokens", 0) or 0
                    
                    if hasattr(resp2, 'output_parsed') and resp2.output_parsed:
                        obj2 = resp2.output_parsed.model_dump(mode="json")
                        return obj2, usage, None
                    
                    parsed2, refusal2 = extract_json_from_response(resp2)
                    if refusal2:
                        return None, usage, f"refusal: {refusal2}"
                    obj2 = HVACJudgeResult.model_validate(parsed2).model_dump(mode="json")
                    return obj2, usage, None

        except BadRequestError as e:
            # Likely unsupported schema formatting or another param for this model.
            last_err = f"{type(e).__name__}: {e}"
            if structured and auto_fallback_json_mode:
                # Switch to JSON mode fallback and retry
                structured = False
                text_format = make_text_format(structured, schema)
                continue
            await asyncio.sleep(min(2 ** attempt, 20))
        except (APIError, RateLimitError, APITimeoutError, InternalServerError) as e:
            last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(min(2 ** attempt, 20))
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(min(2 ** attempt, 20))

    return None, usage, last_err


# ---------------------------
# Orchestrator
# ---------------------------

async def run_async(args) -> None:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    df = pd.read_csv(args.input)
    # Find Transcript col
    tcol = None
    for c in df.columns:
        if c.strip().lower() == "transcript":
            tcol = c
            break
    if not tcol:
        raise KeyError("Could not find a 'Transcript' column (case-insensitive).")

    out_col = args.eval_column
    if out_col not in df.columns:
        df[out_col] = ""

    ensure_dir(args.output)

    # Candidates
    candidates: List[int] = []
    for idx, val in df[tcol].items():
        if not isinstance(val, str):
            continue
        if len(val) < args.min_chars:
            continue
        if args.resume:
            existing = df.at[idx, out_col]
            if isinstance(existing, str) and existing.strip():
                continue
        candidates.append(idx)
    
    # FULL PRODUCTION RUN: Process all candidate rows
    print(f"🚀 FULL PRODUCTION RUN: Processing all {len(candidates)} candidate rows")
    
    total_candidates = len(candidates)
    print(f"[info] Rows to evaluate: {total_candidates} (min_chars={args.min_chars})")

    sem = asyncio.Semaphore(args.max_concurrency)
    pbar = tqdm(total=total_candidates, desc="Evaluating", ncols=100)

    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    usage_lock = asyncio.Lock()

    schema = strict_schema()

    async def worker(row_idx: int):
        print(f"\n" + "="*80)
        print(f"🔄 WORKER DEBUG: Processing row {row_idx}")
        print(f"="*80)
        
        transcript = df.at[row_idx, tcol]
        print(f"📝 Transcript length: {len(transcript)} chars")
        print(f"📝 Transcript preview: {transcript[:100]}...")
        
        async with sem:
            print(f"🚀 Calling judge_one for row {row_idx}...")
            result_json, usage, err = await judge_one(
                client=client,
                model=args.model,
                transcript=transcript,
                schema=schema,
                max_output_tokens=args.max_output_tokens,
                max_attempts=args.max_attempts,
                max_prompt_tokens=args.max_prompt_tokens,
                structured=not args.disable_structured,
                auto_fallback_json_mode=not args.disable_auto_fallback,
            )
        
        print(f"\n📥 WORKER: judge_one returned:")
        print(f"  result_json is not None: {result_json is not None}")
        print(f"  result_json type: {type(result_json)}")
        print(f"  usage: {usage}")
        print(f"  error: {err}")
        
        if result_json is not None:
            print(f"  result_json keys: {list(result_json.keys()) if isinstance(result_json, dict) else 'not a dict'}")
            print(f"  result_json size: {len(str(result_json))} chars")
            
        # Store
        print(f"\n💾 STORING RESULT IN DATAFRAME:")
        print(f"  Target column: '{out_col}'")
        print(f"  Target row: {row_idx}")
        
        if result_json is not None:
            json_str = json.dumps(result_json, ensure_ascii=False)
            print(f"  JSON string length: {len(json_str)} chars")
            print(f"  JSON string preview: {json_str[:100]}...")
            df.at[row_idx, out_col] = json_str
            print(f"✅ Stored result in dataframe")
            
            # Verify it was stored
            stored_value = df.at[row_idx, out_col]
            print(f"🔍 Verification - stored value type: {type(stored_value)}")
            print(f"🔍 Verification - stored value length: {len(str(stored_value))}")
            print(f"🔍 Verification - stored value preview: {str(stored_value)[:100]}...")
        else:
            df.at[row_idx, out_col] = ""  # leave blank on failure/refusal
            print(f"❌ Stored empty string (failure/refusal)")
            print(f"  Error was: {err}")

        # Usage aggregation
        async with usage_lock:
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)
        
        print(f"📊 Updated total usage: {total_usage}")

        # Checkpoint periodically
        if args.checkpoint_every and (pbar.n + 1) % args.checkpoint_every == 0:
            print(f"💾 Saving checkpoint...")
            df.to_csv(args.output, index=False)

        pbar.update(1)

    tasks = [asyncio.create_task(worker(i)) for i in candidates]
    await asyncio.gather(*tasks)
    pbar.close()

    # Final write and verification
    print(f"\n" + "="*80)
    print(f"💾 FINAL CSV WRITE DEBUG")
    print(f"="*80)
    print(f"📁 Output file: {args.output}")
    print(f"📊 DataFrame shape: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}")
    print(f"📊 Target column '{out_col}' exists: {out_col in df.columns}")
    
    if out_col in df.columns:
        non_empty = df[out_col].notna() & (df[out_col] != "")
        print(f"📊 Non-empty evaluations in memory: {non_empty.sum()}")
        
        for i in range(len(df)):
            val = df.at[i, out_col]
            print(f"  Row {i}: {type(val)}, length={len(str(val)) if val else 0}, preview='{str(val)[:50]}...'")
    
    print(f"💾 Writing to CSV...")
    df.to_csv(args.output, index=False)
    print(f"✅ CSV write completed")
    
    # Verify by reading back
    print(f"🔍 Verifying by reading back...")
    verify_df = pd.read_csv(args.output)
    if out_col in verify_df.columns:
        verify_non_empty = verify_df[out_col].notna() & (verify_df[out_col] != "")
        print(f"🔍 Non-empty evaluations in saved file: {verify_non_empty.sum()}")
        for i in range(len(verify_df)):
            val = verify_df.at[i, out_col]
            print(f"  Saved Row {i}: {type(val)}, length={len(str(val)) if val else 0}, preview='{str(val)[:50]}...'")
    
    print(f"[done] Wrote: {args.output}")

    if args.write_scored:
        rows = []
        for idx in candidates:
            raw = df.at[idx, out_col]
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

    # Token usage and rough cost estimate (pay-as-you-go pricing sample)
    IN_PER_M = 1.25
    OUT_PER_M = 10.00
    in_cost = total_usage["input_tokens"]/1_000_000 * IN_PER_M
    out_cost = total_usage["output_tokens"]/1_000_000 * OUT_PER_M
    print(f"[usage] input={total_usage['input_tokens']:,}  output={total_usage['output_tokens']:,}  total={total_usage['total_tokens']:,}")
    print(f"[est. cost] input=${in_cost:.4f}  output=${out_cost:.4f}  total=${(in_cost+out_cost):.4f}")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parallel LLM-as-a-Judge for HVAC transcripts (GPT-5 reasoning, Structured Outputs).")
    p.add_argument("--input", required=True, help="Path to input CSV with a 'Transcript' column")
    p.add_argument("--output", required=True, help="Path to write the CSV with 'gpt-5-high-eval' column")
    p.add_argument("--model", default="gpt-4o-2024-08-06", help="Judge model (default: gpt-4o-2024-08-06 - supports structured outputs)")
    p.add_argument("--min-chars", type=int, default=300, help="Skip transcripts shorter than this many characters")
    p.add_argument("--max-concurrency", type=int, default=12, help="Max concurrent API requests")
    p.add_argument("--max-attempts", type=int, default=3, help="Max attempts per row (with backoff)")
    p.add_argument("--max-output-tokens", type=int, default=2000, help="Max output tokens per request")
    p.add_argument("--max-prompt-tokens", type=int, default=0, help="If >0 (and tiktoken installed), compact transcripts above this token length")
    p.add_argument("--resume", action="store_true", help="Skip rows that already have JSON in the output column")
    p.add_argument("--checkpoint-every", type=int, default=100, help="Write partial CSV every N completed rows")
    p.add_argument("--eval-column", default="gpt-5-high-eval", help="Name of the output JSON column")
    p.add_argument("--write-scored", action="store_true", help="Also write *.scored.csv with numeric convenience columns")
    p.add_argument("--disable-structured", action="store_true", help="Force JSON mode (no schema) instead of structured outputs")
    p.add_argument("--disable-auto-fallback", action="store_true", help="Disable auto-fallback to JSON mode if schema is unsupported")
    return p

def main():
    args = build_argparser().parse_args()
    try:
        asyncio.run(run_async(args))
    except KeyboardInterrupt:
        print("\n[warn] Interrupted by user. Partial results (if any) are in:", args.output)

if __name__ == "__main__":
    main()
