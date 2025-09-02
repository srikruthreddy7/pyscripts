import pandas as pd
import re
from pathlib import Path
import vapi
import time
import logging
from typing import Optional, List, Dict, Any
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VAPIClient:
    def __init__(self, api_key: str, max_retries: int = 3, timeout: int = 60):
        self.client = vapi.Vapi(
            token=api_key,
            # Configure httpx client with proper timeouts and connection limits
            httpx_client=httpx.Client(
                timeout=httpx.Timeout(timeout),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                # Add retry transport to handle connection issues
                transport=httpx.HTTPTransport(retries=max_retries)
            )
        )
        self.max_retries = max_retries
    
    def fetch_calls_with_retry(self, **params) -> List[Dict[str, Any]]:
        """Fetch calls with exponential backoff retry logic."""
        all_calls = []
        retries = 0
        
        while retries <= self.max_retries:
            try:
                logger.info(f"Fetching calls (attempt {retries + 1}/{self.max_retries + 1})")
                
                # Use pagination to avoid large responses that might timeout
                page_size = params.get('limit', 100)
                params['limit'] = min(page_size, 100)  # Limit to 100 per request
                
                response = self.client.calls.list(**params)
                
                if hasattr(response, 'data'):
                    all_calls.extend(response.data)
                else:
                    all_calls.extend(response)
                
                logger.info(f"Successfully fetched {len(all_calls)} calls")
                return all_calls
                
            except (httpx.RemoteProtocolError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                retries += 1
                if retries > self.max_retries:
                    logger.error(f"Max retries exceeded. Last error: {e}")
                    raise
                
                # Exponential backoff: 2^retry_count seconds
                wait_time = 2 ** retries
                logger.warning(f"HTTP error occurred: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
        
        return all_calls

def find_transcript_column(columns):
    """Find transcript column in DataFrame."""
    for c in columns:
        if c.strip().lower() == "transcript":
            return c
    for c in columns:
        if "transcript" in c.strip().lower():
            return c
    return None

# HVAC and test keywords for classification
HVAC_KEYWORDS = [
    "thermostat","condenser","evaporator","compressor","refrigerant","txv","tev","capillary",
    "air handler","condenser fan","blower","ecm","psc","capacitor","contactor","heat pump",
    "split system","package unit","rooftop unit","rtu","vav","ahu","furnace","inducer","ignitor","igniter",
    "flame sensor","gas valve","heat exchanger","defrost","aux heat","auxiliary heat","reversing valve","o/b wire",
    "suction","liquid line","superheat","subcooling","static pressure","cfm","duct","plenum","damper","filter",
    "condensate","drain pan","float switch","wet switch","trap","condensate pump","wiring diagram",
    "transformer","sequencer","heat strips","seer","tonnage","micron","vacuum","recovery","nitrogen","brazing",
    "flare","schrader","service valve","manifold","gauge","saturation","ambient","supply","return","delta t",
    "txv bulb","sensing bulb","clamp meter","amp draw","amperage","ohms","continuity","high pressure switch",
    "low pressure switch","lockout","board","evaporator coil","coil","fins","shroud","fan blade","hard start","run cap",
    "trane","carrier","bryant","payne","lennox","goodman","amana","daikin","mitsubishi","fujitsu","rheem","ruud",
    "york","heil","icp","american standard","bosch","gree","lg"
]

HVAC_PATTERNS = [
    r"\bR-?410A\b", r"\bR-?22\b", r"\bPSI\b", r"\b\d{2,3}\s?psi\b",
    r"\b\d{2,3}\s?(?:°F|F)\b", r"\b\d{1,3}\s?amps?\b", r"\b\d{1,2}\.?\d?\s?(?:µF|uF)\b",
    r"\b\d(?:\.\d)?\s?ton\b", r"\bSEER\s?\d{1,2}\b",
    r"\b\d{2}x\d{2}x\d{1,2}\b", r"\bmodel\b", r"\bserial\b", r"\bsuper\s?heat\b", r"\bsub\s?cool(ing)?\b"
]

TEST_KEYWORDS = [
    "test","testing","dummy","qwerty","asdf","lorem","ipsum","hello world","prompt","system prompt","jailbreak",
    "ignore previous","as an ai language model","dataset","fine-tune","finetune","training data","evaluation","benchmark",
    "knowledge cutoff","chatgpt","gpt","claude","llama","mistral","openai","anthropic","llm","prompt injection",
    "roleplay","pretend","act as","developer","dev","debug","logs","context window","temperature parameter",
    "system message","assistant message","user message","tool call","json schema","yaml","markdown","api key",
    "write code","python","javascript","java","react","next.js","docker","kubernetes","terraform"
]

code_like_patterns = [r"```", r"\{\s*\".*\"\s*:\s*.*\}", r"\bdef\s+\w+\(", r"\bclass\s+\w+"]

def word_count(s): 
    return len(re.findall(r"\w+", s))

def score_transcript(text):
    """Score transcript for HVAC relevance vs test content."""
    if not isinstance(text, str):
        text = ""
    low = text.lower()

    user_turns = re.findall(r"(?im)^\s*user\s*:\s*(.*)$", text)
    ai_turns = re.findall(r"(?im)^\s*ai\s*:\s*(.*)$", text)

    short_user = sum(1 for u in user_turns if word_count(u) < 4)
    user_count = max(len(user_turns), 1)
    short_ratio = short_user / user_count

    hvac_hits = [kw for kw in HVAC_KEYWORDS if kw in low]
    hvac_score = len(set(hvac_hits))

    hvac_pat_score = 0
    hvac_pat_examples = []
    for pat in HVAC_PATTERNS:
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m:
            hvac_pat_score += 1
            hvac_pat_examples.append(m[0] if isinstance(m[0], str) else str(m[0]))

    hvac_total = hvac_score + hvac_pat_score

    test_hits = [kw for kw in TEST_KEYWORDS if kw in low]
    test_score = len(set(test_hits))

    for pat in code_like_patterns:
        if re.search(pat, text, flags=re.IGNORECASE | re.DOTALL):
            test_score += 1
            break

    if short_ratio > 0.7 and hvac_total < 2:
        test_score += 1

    if hvac_total >= 3 and test_score == 0:
        label = "legit"
    elif hvac_total >= 4 and test_score <= 1:
        label = "legit"
    elif (test_score >= 2 and hvac_total < 3) or (hvac_total == 0):
        label = "bad"
    else:
        label = "bad"

    reason = {
        "hvac_keyword_hits": min(10, len(set(hvac_hits))),
        "hvac_pattern_hits": hvac_pat_score,
        "test_keyword_hits": min(10, len(set(test_hits))),
        "short_user_ratio": round(short_ratio, 2),
        "user_turns": len(user_turns),
        "ai_turns": len(ai_turns),
        "example_hvac_pattern": hvac_pat_examples[:3],
    }
    return label, hvac_total, test_score, reason

def main():
    # Option 1: Process existing CSV file
    input_csv = r"/Users/Srikruth/Downloads/calls-export-737ed8f4-0ad9-4dbe-9bb3-214c04da2c76-2025-08-19-19-05-20-user5plus.csv"
    
    try:
        # Check if CSV exists
        orig = Path(input_csv).expanduser().resolve()
        if orig.exists():
            logger.info(f"Processing existing CSV: {orig}")
            df_full = pd.read_csv(orig, low_memory=False)
        else:
            # Option 2: Fetch from VAPI API
            logger.info("CSV not found. Fetching from VAPI API...")
            
            # You'll need to set your VAPI API key
            api_key = "your_vapi_api_key_here"  # Replace with actual API key
            if api_key == "your_vapi_api_key_here":
                raise ValueError("Please set your VAPI API key")
            
            vapi_client = VAPIClient(api_key, max_retries=3, timeout=120)
            
            # Fetch calls with retry logic
            params = {
                'limit': 100,  # Start with smaller batches
                # Add other filters as needed
            }
            
            calls_data = vapi_client.fetch_calls_with_retry(**params)
            
            # Convert to DataFrame
            df_full = pd.DataFrame(calls_data)
            
            # Save fetched data
            orig = Path("fetched_calls.csv")
            df_full.to_csv(orig, index=False)
            logger.info(f"Saved fetched data to: {orig}")

        # Find transcript column
        tcol = find_transcript_column(df_full.columns)
        if tcol is None:
            raise ValueError(f"Transcript column not found. Available columns: {list(df_full.columns)}")

        # Filter for calls with 5+ user turns
        user_turn_counts = (
            df_full[tcol]
            .fillna("")
            .astype(str)
            .str.count(r"(?im)^\s*user\s*:")
        )
        mask_5plus = user_turn_counts >= 5
        df = df_full[mask_5plus].copy()

        logger.info(f"Found {len(df)} calls with 5+ user turns")

        # Score transcripts
        labels, hvac_scores, test_scores, reasons = [], [], [], []
        for txt in df[tcol].fillna("").astype(str):
            label, hs, ts, reason = score_transcript(txt)
            labels.append(label)
            hvac_scores.append(hs)
            test_scores.append(ts)
            reasons.append(reason)

        # Create output DataFrame
        df_out = df.copy()
        df_out["FineTuneLabel"] = labels
        df_out["HVAC_Score"] = hvac_scores
        df_out["Test_Score"] = test_scores
        df_out["Reason"] = [str(r) for r in reasons]

        # Save results
        prefix = orig.with_suffix("").name
        out_legit = orig.with_name(f"{prefix}-finetune_legit.csv")
        out_bad = orig.with_name(f"{prefix}-finetune_bad.csv")

        df_out[df_out["FineTuneLabel"] == "legit"].to_csv(out_legit, index=False)
        df_out[df_out["FineTuneLabel"] == "bad"].to_csv(out_bad, index=False)

        results = {
            "source_file": str(orig),
            "transcript_column": tcol,
            "input_rows_with_5+_user_turns": int(len(df)),
            "legit_rows": int((df_out["FineTuneLabel"] == "legit").sum()),
            "bad_rows": int((df_out["FineTuneLabel"] == "bad").sum()),
            "legit_output_file": str(out_legit),
            "bad_output_file": str(out_bad),
        }
        
        print(results)
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
