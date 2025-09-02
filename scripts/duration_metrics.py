#!/usr/bin/env python3
import os
import sys
import re
import json
import math
import statistics
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*_a, **_k):
        return False


def parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def main() -> None:
    load_dotenv()
    uri = os.getenv("MONGODB_URI") or os.getenv("MONGODB_CONNECTION_STRING") or os.getenv("MONGO_URI")
    db_name = os.getenv("MONGODB_DB_NAME") or os.getenv("MONGODB_DB") or "LinkProDB"
    coll_name = os.getenv("MONGODB_COLLECTION") or "linkprosessions"
    if not uri:
        print("Missing MONGODB_URI", file=sys.stderr)
        sys.exit(2)

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
    except ConnectionFailure as e:
        print(f"Connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    db = client[db_name]
    col = db[coll_name]

    # Build filter from envs
    filt = {}
    raw = os.getenv("QUERY_FILTER")
    if raw:
        try:
            base = json.loads(raw)
            if isinstance(base, dict):
                filt.update(base)
        except Exception:
            pass

    since_iso = os.getenv("SINCE_ISO") or "2024-01-22T09:02:32.000Z"
    prefix = os.getenv("VIDEO_PREFIX") or "https://linkproaccstorageadmin.blob.core.windows.net"
    support_not_null = (os.getenv("SUPPORT_NOT_NULL") or "1") not in ("0", "false", "False")

    try:
        cutoff = parse_iso(since_iso)
        created = filt.get("createdAt") or {}
        created["$gte"] = cutoff
        filt["createdAt"] = created
    except Exception as e:
        print(f"Invalid SINCE_ISO: {e}", file=sys.stderr)
        sys.exit(2)

    video = filt.get("videoUrl") or {}
    video["$regex"] = "^" + re.escape(prefix)
    filt["videoUrl"] = video

    if support_not_null:
        filt["supportIdMember"] = {"$ne": None}

    # Fetch durations and createdAt
    try:
        cursor = col.find(filt, {"videoDuration": 1, "createdAt": 1})
    except Exception as e:
        print(f"Query error: {e}", file=sys.stderr)
        sys.exit(1)

    durations: list[float] = []
    per_month: dict[str, float] = {}
    for doc in cursor:
        dur = doc.get("videoDuration", 0)
        try:
            duration_s = float(dur)
        except Exception:
            continue
        durations.append(duration_s)
        created_at = doc.get("createdAt")
        if hasattr(created_at, "strftime"):
            ym = created_at.strftime("%Y-%m")
            per_month[ym] = per_month.get(ym, 0.0) + duration_s

    total_s = sum(durations)
    mean_s = (total_s / len(durations)) if durations else 0.0
    median_s = statistics.median(durations) if durations else 0.0

    print(f"COUNT_MATCHED: {len(durations)}")
    print(f"TOTAL_SECONDS: {int(total_s)}")
    print(f"TOTAL_HOURS: {round(total_s / 3600.0, 2)}")
    print(f"MEAN_SECONDS: {round(mean_s, 2)}")
    print(f"MEDIAN_SECONDS: {round(median_s, 2)}")
    print("TOTAL_SECONDS_PER_MONTH:")
    for ym in sorted(per_month):
        print(f"  {ym}: {int(per_month[ym])}")


if __name__ == "__main__":
    main()


