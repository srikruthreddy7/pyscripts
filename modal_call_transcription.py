#!/usr/bin/env python3
"""
Modal version of call_transcription_since_nov2024.py

Environment Variables Used (set in Modal secrets):
===========================================

MongoDB Configuration:
- MONGODB_URI (or MONGODB_CONNECTION_STRING or MONGO_URI): MongoDB connection string
- MONGODB_DB_NAME (or MONGODB_DB): Database name (defaults to "LinkProDB")
- MONGODB_COLLECTION: Collection name (defaults to "linkprosessions")

Azure Function Configuration:
- AZURE_FUNCTION_URL: URL of the Azure Function to call for transcription

Processing Configuration:
- LIMIT: Maximum number of sessions to process (0 = unlimited)
- DRY_RUN: Enable dry run mode (1=true, 0=false, defaults to true)
- SLEEP_BETWEEN_CALLS_S: Seconds to sleep between API calls (defaults to 0.0)
- CLAIM: Enable session claiming (1=true, 0=false, defaults to true)
- SKIP_IF_ALREADY_CLAIMED: Skip already claimed sessions (1=true, 0=false, defaults to true)
- ADD_EGRESS_ID_PREFIX: Prefix for egress IDs (defaults to "manual-backfill")
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone
from typing import Optional

import modal
import requests
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pymongo.collection import ReturnDocument

# Modal setup
app = modal.App("call-transcription-processor")

# Image with required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "pymongo>=4.6.0",
        "requests>=2.32.3"
    )
)

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Filters:
# - createdAt >= 2024-10-01T00:00:00Z AND < 2025-03-01T00:00:00Z (Oct 1st, 2024 to Mar 1st, 2025, not included)
# - videoUrl exists AND starts with https://linkproaccstorageadmin.blob.core.windows.net
# - videoDuration >= 90
# - supportIdMember != null

START_DATE_ISO = "2024-10-01T00:00:00.000Z"
END_DATE_ISO = "2025-03-01T00:00:00.000Z"
AZURE_PREFIX = "https://linkproaccstorageadmin.blob.core.windows.net"


def iso_utc(dt: datetime) -> str:
	if dt.tzinfo is None:
		dt = dt.replace(tzinfo=timezone.utc)
	return dt.astimezone(timezone.utc).isoformat()


def parse_iso(value: str) -> datetime:
	return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def get_mongo_collection() -> any:
	logger.info("🔌 Establishing MongoDB connection...")
	uri = os.getenv("MONGODB_URI") or os.getenv("MONGODB_CONNECTION_STRING") or os.getenv("MONGO_URI")
	db_name = os.getenv("MONGODB_DB_NAME") or os.getenv("MONGODB_DB") or "LinkProDB"
	coll_name = os.getenv("MONGODB_COLLECTION") or "linkprosessions"

	logger.info(f"📋 Connection details: {db_name}.{coll_name}")

	if not uri:
		logger.error("❌ Missing MONGODB_URI environment variable")
		sys.exit(2)

	logger.info("🔗 Connecting to MongoDB...")
	client = MongoClient(uri, serverSelectionTimeoutMS=5000)
	logger.info("🏥 Testing connection with ping...")
	client.admin.command("ping")
	logger.info("✅ MongoDB connection established successfully")
	return client[db_name][coll_name]


def build_filter() -> dict:
	logger.info("🏗️ Building session filter criteria...")
	start_date = parse_iso(START_DATE_ISO)
	end_date = parse_iso(END_DATE_ISO)
	filter_dict = {
		"createdAt": {"$gte": start_date, "$lt": end_date},
		"videoUrl": {"$exists": True, "$regex": f"^{AZURE_PREFIX.replace('.', '\\.')}"},
		"videoDuration": {"$gte": 90},
		"supportIdMember": {"$ne": None},
	}
	logger.info(f"🔍 Filter criteria: {json.dumps(filter_dict, indent=2, default=str)}")
	return filter_dict


def get_func_url() -> str:
	logger.info("🌐 Getting Azure Function URL...")
	url = os.getenv("AZURE_FUNCTION_URL")
	if not url:
		logger.error("❌ Missing AZURE_FUNCTION_URL environment variable")
		sys.exit(2)
	logger.info(f"🔗 Azure Function URL: {url}")
	return url


def call_function(func_url: str, video_url: str, session_id: str, timeout_s: int = 300) -> requests.Response:
	logger.info(f"📡 Preparing API call to Azure Function...")
	logger.info(f"🎯 Target URL: {func_url}")
	logger.info(f"📹 Video URL: {video_url}")
	logger.info(f"🆔 Session ID: {session_id}")
	logger.info(f"⏱️ Timeout: {timeout_s}s")

	payload = {"videoUrl": video_url, "sessionId": session_id}
	logger.info(f"📦 Request payload: {json.dumps(payload)}")
	logger.info(f"📏 Payload size: {len(json.dumps(payload))} bytes")

	logger.info("🚀 Sending request to Azure Function...")
	response = requests.post(func_url, json=payload, timeout=timeout_s)

	logger.info(f"📥 Response received:")
	logger.info(f"📊 Status Code: {response.status_code}")
	logger.info(f"📏 Response size: {len(response.text)} bytes")
	logger.info(f"📝 Response content: {response.text[:500]}{'...' if len(response.text) > 500 else ''}")

	return response


def claim_session(collection, doc: dict, egress_prefix: str) -> dict | None:
	"""Simulate API server idempotency claim on LinkProSession.

	- Only claim if no transcriptionRequestedAt exists yet
	- Set transcriptionRequestedAt = now
	- Optionally add a synthetic egress id to processedEgressIds
	"""
	session_id = doc.get("sessionId") or str(doc.get("_id"))
	logger.info(f"🔒 Starting claim process for session: {session_id}")

	# Build filter for atomic claim
	filt = {"_id": doc.get("_id"), "transcriptionRequestedAt": {"$exists": False}}
	logger.info(f"🎯 Claim filter: {json.dumps(filt, default=str)}")

	# Build update document
	update: dict = {"$set": {"transcriptionRequestedAt": datetime.now(timezone.utc)}}
	logger.info(f"📝 Base update: {json.dumps(update, default=str)}")

	if egress_prefix:
		egress_id = f"{egress_prefix}-{int(time.time())}"
		update["$addToSet"] = {"processedEgressIds": egress_id}
		logger.info(f"📋 Adding egress ID: {egress_id}")

	logger.info("💾 Executing atomic claim update...")
	result = collection.find_one_and_update(
		filt,
		update,
		return_document=ReturnDocument.AFTER,
	)

	if result:
		logger.info(f"✅ Claim SUCCESSFUL for session: {session_id}")
		logger.info(f"📊 Claim result: transcriptionRequestedAt={result.get('transcriptionRequestedAt')}")
		if result.get('processedEgressIds'):
			logger.info(f"📋 Egress IDs: {result.get('processedEgressIds')}")
	else:
		logger.warning(f"⚠️ Claim FAILED (already claimed or not found): {session_id}")

	return result


def process_transcriptions():
	"""Main transcription processing function."""
	logger.info("🚀 Starting transcription processing script...")
	logger.info("=" * 60)
	
	collection = get_mongo_collection()
	filt = build_filter()

	# Configuration logging
	limit = int(os.getenv("LIMIT") or "0")  # 0 = no limit
	dry_run = (os.getenv("DRY_RUN") or "1") not in ("0", "false", "False")
	sleep_s = float(os.getenv("SLEEP_BETWEEN_CALLS_S") or "0.0")
	func_url = get_func_url()
	claim_enabled = (os.getenv("CLAIM") or "1") not in ("0", "false", "False")
	skip_if_claimed = (os.getenv("SKIP_IF_ALREADY_CLAIMED") or "1") not in ("0", "false", "False")
	egress_prefix = os.getenv("ADD_EGRESS_ID_PREFIX") or "manual-backfill"

	logger.info("⚙️ Configuration:")
	logger.info(f"  📏 LIMIT: {limit if limit > 0 else 'UNLIMITED'}")
	logger.info(f"  🏃 DRY_RUN: {dry_run}")
	logger.info(f"  😴 SLEEP_BETWEEN_CALLS_S: {sleep_s}")
	logger.info(f"  🔒 CLAIM_ENABLED: {claim_enabled}")
	logger.info(f"  ⏭️ SKIP_IF_ALREADY_CLAIMED: {skip_if_claimed}")
	logger.info(f"  🏷️ EGRESS_PREFIX: {egress_prefix}")

	# Session counting
	logger.info("🔢 Counting matching sessions...")
	count = collection.count_documents(filt)
	logger.info(f"📊 Total sessions matching criteria: {count}")

	# Query building
	logger.info("📋 Building session query...")
	query_fields = {"_id": 1, "sessionId": 1, "videoUrl": 1, "videoDuration": 1, "createdAt": 1, "transcriptionRequestedAt": 1}
	logger.info(f"📝 Query fields: {list(query_fields.keys())}")

	# Batch processing to avoid cursor timeouts
	batch_size = 50  # Process 50 sessions at a time
	total_processed = 0
	total_sent = 0

	logger.info(f"📦 Using batch size: {batch_size} sessions per batch")

	while True:
		logger.info(f"🔄 Starting batch {total_processed // batch_size + 1}...")

		# Create new cursor for each batch to avoid timeouts
		cursor = collection.find(filt, query_fields).sort("createdAt", 1).skip(total_processed)

		if limit > 0 and total_processed >= limit:
			logger.info(f"🎯 Reached limit of {limit} sessions")
			break

		# Apply limit if specified
		if limit > 0:
			remaining = limit - total_processed
			cursor = cursor.limit(min(batch_size, remaining))
		else:
			cursor = cursor.limit(batch_size)

		batch_docs = list(cursor)  # Convert to list to release cursor immediately

		if not batch_docs:
			logger.info("🏁 No more sessions to process")
			break

		logger.info(f"📊 Processing batch of {len(batch_docs)} sessions...")

		sent = 0
		for doc in batch_docs:
			logger.info("-" * 50)
			logger.info("🎬 Processing new session...")

			_session_id = doc.get("sessionId") or str(doc.get("_id"))
			_video_url = doc.get("videoUrl")
			_dur = doc.get("videoDuration")
			_created = doc.get("createdAt")
			created_iso = iso_utc(_created) if isinstance(_created, datetime) else str(_created)
			already_claimed = bool(doc.get("transcriptionRequestedAt"))

			logger.info(f"🆔 Session ID: {_session_id}")
			logger.info(f"📹 Video URL: {_video_url}")
			logger.info(f"⏱️ Duration: {_dur}s")
			logger.info(f"📅 Created: {created_iso}")
			logger.info(f"🔖 Already claimed: {already_claimed}")

			# Simulate idempotency claim like the webhook would
			if claim_enabled:
				logger.info("🔒 Claiming phase...")
				if already_claimed:
					logger.info(f"⚠️ Session already claimed, skipping claim attempt")
					if skip_if_claimed:
						logger.info(f"⏭️ Skipping session due to existing claim")
						continue
				else:
					if dry_run:
						logger.info(f"🏃 DRY RUN: Would claim session {_session_id}")
					else:
						logger.info(f"🔐 Attempting to claim session {_session_id}...")
						try:
							claimed_doc = claim_session(collection, doc, egress_prefix)
							if not claimed_doc:
								logger.warning(f"❌ Claim not applied (race condition or already claimed): {_session_id}")
						except Exception as e:
							logger.error(f"💥 Claim error for session {_session_id}: {e}")
			else:
				logger.info("🔓 Claiming disabled, proceeding without claim")

			if dry_run:
				logger.info(f"🏃 DRY RUN: Would call Azure function for session {_session_id}")
			else:
				logger.info(f"📞 Calling Azure function for session {_session_id}...")
				try:
					resp = call_function(func_url, _video_url, _session_id)
					if resp.status_code in (200, 202):
						logger.info(f"✅ API call successful for session {_session_id}")
					else:
						logger.warning(f"⚠️ API call returned status {resp.status_code} for session {_session_id}")
					sent += 1
					if sleep_s > 0:
						logger.info(f"😴 Sleeping for {sleep_s} seconds...")
						time.sleep(sleep_s)
				except requests.RequestException as e:
					logger.error(f"💥 API call failed for session {_session_id}: {e}")

		# Update totals and log batch completion
		total_processed += len(batch_docs)
		total_sent += sent

		logger.info(f"✅ Batch {total_processed // batch_size + 1} completed!")
		logger.info(f"📊 Batch processed: {len(batch_docs)} sessions")
		logger.info(f"📡 Batch sent: {sent} API calls")
		logger.info(f"📈 Running total: {total_processed} sessions processed, {total_sent} sent")
		logger.info("-" * 60)

	logger.info("=" * 60)
	logger.info(f"🏁 All processing complete!")
	logger.info(f"📊 Final totals:")
	logger.info(f"  📈 Total sessions processed: {total_processed}")
	logger.info(f"  📡 Total API calls sent: {total_sent}")
	logger.info(f"  🏃 Dry run mode: {dry_run}")
	logger.info("=" * 60)

	return {
		"total_processed": total_processed,
		"total_sent": total_sent,
		"dry_run": dry_run
	}


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("call_script")],
    timeout=24 * 60 * 60,  # 24 hours timeout for long-running batch jobs
    memory=2048,  # 2GB RAM
)
def run_transcription_processor():
    """Modal function to run the transcription processing."""
    return process_transcriptions()
