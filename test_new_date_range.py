#!/usr/bin/env python3
"""
Test the new date range (Oct 1st, 2024 to Mar 1st, 2025)
"""

import os
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
	
	client = MongoClient(uri, serverSelectionTimeoutMS=5000)
	collection = client[db_name][coll_name]
	
	print("✅ Connected to MongoDB")
	
	# New date range: Oct 1st, 2024 to Mar 1st, 2025 (not included)
	start_date = parse_iso("2024-10-01T00:00:00.000Z")
	end_date = parse_iso("2025-03-01T00:00:00.000Z")
	azure_prefix = "https://linkproaccstorageadmin.blob.core.windows.net"
	
	new_filter = {
		"createdAt": {"$gte": start_date, "$lt": end_date},
		"videoUrl": {"$exists": True, "$regex": f"^{azure_prefix.replace('.', '\\.')}"},
		"videoDuration": {"$gte": 90},
		"supportIdMember": {"$ne": None},
	}
	
	count = collection.count_documents(new_filter)
	print(f"📊 Sessions matching new filter (Oct 1st, 2024 to Mar 1st, 2025): {count}")
	
	# Show date breakdown
	print(f"\n📅 Date range: {start_date} to {end_date}")
	print(f"📅 In human terms: October 1st, 2024 to March 1st, 2025 (not included)")
	
	# Sample a few documents
	if count > 0:
		print(f"\n📝 Sample sessions:")
		sample_docs = list(collection.find(new_filter).sort("createdAt", 1).limit(5))
		for i, doc in enumerate(sample_docs):
			print(f"  {i+1}. {doc.get('sessionId')} - Created: {doc.get('createdAt')}")

if __name__ == "__main__":
	main()
