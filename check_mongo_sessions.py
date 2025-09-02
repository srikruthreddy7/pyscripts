#!/usr/bin/env python3
"""
Quick script to check MongoDB sessions directly
"""

import os
import json
from datetime import datetime, timezone
from pymongo import MongoClient

try:
	from dotenv import load_dotenv
	load_dotenv()
except Exception:
	pass

def parse_iso(value: str) -> datetime:
	return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)

def main():
	print("🔌 Connecting to MongoDB...")
	
	uri = os.getenv("MONGODB_URI") or os.getenv("MONGODB_CONNECTION_STRING") or os.getenv("MONGO_URI")
	db_name = os.getenv("MONGODB_DB_NAME") or os.getenv("MONGODB_DB") or "LinkProDB"
	coll_name = os.getenv("MONGODB_COLLECTION") or "linkprosessions"
	
	if not uri:
		print("❌ Missing MONGODB_URI environment variable")
		return
	
	client = MongoClient(uri, serverSelectionTimeoutMS=5000)
	client.admin.command("ping")
	collection = client[db_name][coll_name]
	
	print(f"✅ Connected to {db_name}.{coll_name}")
	
	# First, check total documents
	total_docs = collection.count_documents({})
	print(f"📊 Total documents in collection: {total_docs}")
	
	# Check recent documents
	recent_filter = {"createdAt": {"$gte": datetime(2024, 1, 1, tzinfo=timezone.utc)}}
	recent_count = collection.count_documents(recent_filter)
	print(f"📊 Documents since 2024-01-01: {recent_count}")
	
	# Check with video URL filter
	video_filter = {"videoUrl": {"$exists": True}}
	video_count = collection.count_documents(video_filter)
	print(f"📊 Documents with videoUrl: {video_count}")
	
	# Check with Azure blob filter
	azure_prefix = "https://linkproaccstorageadmin.blob.core.windows.net"
	azure_filter = {"videoUrl": {"$exists": True, "$regex": f"^{azure_prefix.replace('.', '\\\\.')}"}}
	azure_count = collection.count_documents(azure_filter)
	print(f"📊 Documents with Azure blob videoUrl: {azure_count}")
	
	# Check duration filter
	duration_filter = {"videoDuration": {"$gte": 90}}
	duration_count = collection.count_documents(duration_filter)
	print(f"📊 Documents with videoDuration >= 90: {duration_count}")
	
	# Check supportIdMember filter
	support_filter = {"supportIdMember": {"$ne": None}}
	support_count = collection.count_documents(support_filter)
	print(f"📊 Documents with supportIdMember != null: {support_count}")
	
	# Now check the original strict filter
	cutoff_iso = "2025-03-01T00:00:00.000Z"
	cutoff = parse_iso(cutoff_iso)
	strict_filter = {
		"createdAt": {"$gte": cutoff},
		"videoUrl": {"$exists": True, "$regex": f"^{azure_prefix.replace('.', '\\\\.')}"},
		"videoDuration": {"$gte": 90},
		"supportIdMember": {"$ne": None},
	}
	strict_count = collection.count_documents(strict_filter)
	print(f"📊 Documents matching strict filter (createdAt >= 2025-03-01): {strict_count}")
	
	# Let's try a more reasonable date - November 2024
	nov_filter = {
		"createdAt": {"$gte": datetime(2024, 11, 1, tzinfo=timezone.utc)},
		"videoUrl": {"$exists": True, "$regex": f"^{azure_prefix.replace('.', '\\\\.')}"},
		"videoDuration": {"$gte": 90},
		"supportIdMember": {"$ne": None},
	}
	nov_count = collection.count_documents(nov_filter)
	print(f"📊 Documents matching filter (createdAt >= 2024-11-01): {nov_count}")
	
	# Sample a few recent documents to see their structure
	print("\n📝 Sample recent documents:")
	recent_docs = list(collection.find({}).sort("createdAt", -1).limit(3))
	for i, doc in enumerate(recent_docs):
		print(f"\nDocument {i+1}:")
		print(f"  _id: {doc.get('_id')}")
		print(f"  sessionId: {doc.get('sessionId')}")
		print(f"  createdAt: {doc.get('createdAt')}")
		print(f"  videoUrl: {doc.get('videoUrl', 'Missing')[:100]}..." if doc.get('videoUrl') else "  videoUrl: None")
		print(f"  videoDuration: {doc.get('videoDuration')}")
		print(f"  supportIdMember: {doc.get('supportIdMember')}")
		print(f"  transcriptionRequestedAt: {doc.get('transcriptionRequestedAt')}")

if __name__ == "__main__":
	main()
