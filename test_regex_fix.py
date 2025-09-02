#!/usr/bin/env python3
"""
Test and fix the regex pattern for Azure blob URLs
"""

import os
import re
from pymongo import MongoClient

try:
	from dotenv import load_dotenv
	load_dotenv()
except Exception:
	pass

def main():
	print("🔌 Connecting to MongoDB...")
	
	uri = os.getenv("MONGODB_URI") or os.getenv("MONGODB_CONNECTION_STRING") or os.getenv("MONGO_URI")
	db_name = os.getenv("MONGODB_DB_NAME") or os.getenv("MONGODB_DB") or "LinkProDB"
	coll_name = os.getenv("MONGODB_COLLECTION") or "linkprosessions"
	
	client = MongoClient(uri, serverSelectionTimeoutMS=5000)
	collection = client[db_name][coll_name]
	
	print("✅ Connected to MongoDB")
	
	# Get a sample URL from the database
	sample_doc = collection.find_one({"videoUrl": {"$exists": True, "$ne": None}})
	if sample_doc and sample_doc.get("videoUrl"):
		sample_url = sample_doc.get("videoUrl")
		print(f"📝 Sample URL from DB: {sample_url}")
		
		azure_prefix = "https://linkproaccstorageadmin.blob.core.windows.net"
		
		# Test different regex patterns
		patterns = [
			f"^{azure_prefix.replace('.', '\\\\.')}",  # Original pattern
			f"^{azure_prefix.replace('.', '\\.')}",    # Single escape
			f"^{azure_prefix}",                        # No escaping
			f"^https://linkproaccstorageadmin\\.blob\\.core\\.windows\\.net",  # Manual escaping
		]
		
		for i, pattern in enumerate(patterns):
			print(f"\n🧪 Testing pattern {i+1}: {pattern}")
			
			# Test with Python re module
			python_match = bool(re.match(pattern, sample_url))
			print(f"   Python re.match: {python_match}")
			
			# Test with MongoDB query
			mongo_count = collection.count_documents({"videoUrl": {"$regex": pattern}})
			print(f"   MongoDB count: {mongo_count}")
			
		# Now test the working pattern with all filters
		working_pattern = f"^{azure_prefix}"  # Simple no-escape version
		print(f"\n🎯 Testing full filter with working pattern...")
		
		from datetime import datetime, timezone
		cutoff = datetime(2025, 3, 1, tzinfo=timezone.utc)
		
		full_filter = {
			"createdAt": {"$gte": cutoff},
			"videoUrl": {"$exists": True, "$regex": working_pattern},
			"videoDuration": {"$gte": 90},
			"supportIdMember": {"$ne": None},
		}
		
		full_count = collection.count_documents(full_filter)
		print(f"📊 Full filter count: {full_count}")
		
		# If still 0, try more recent date
		recent_cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
		recent_filter = {
			"createdAt": {"$gte": recent_cutoff},
			"videoUrl": {"$exists": True, "$regex": working_pattern},
			"videoDuration": {"$gte": 90},
			"supportIdMember": {"$ne": None},
		}
		
		recent_count = collection.count_documents(recent_filter)
		print(f"📊 Recent filter count (since 2024): {recent_count}")
		
		if recent_count > 0:
			print("✅ Found sessions! The date filter might be too restrictive.")
			# Show a few matching sessions
			matches = list(collection.find(recent_filter).limit(3))
			for doc in matches:
				print(f"   📄 Session: {doc.get('sessionId')} - Created: {doc.get('createdAt')}")

if __name__ == "__main__":
	main()
