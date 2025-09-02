#!/usr/bin/env python3
"""
Calculate total duration of sessions in both date ranges
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

def format_duration(total_seconds: float) -> str:
	"""Convert seconds to hours and minutes"""
	hours = int(total_seconds // 3600)
	minutes = int((total_seconds % 3600) // 60)
	return f"{hours}h {minutes}m"

def main():
	print("🔌 Connecting to MongoDB...")
	
	uri = os.getenv("MONGODB_URI") or os.getenv("MONGODB_CONNECTION_STRING") or os.getenv("MONGO_URI")
	db_name = os.getenv("MONGODB_DB_NAME") or os.getenv("MONGODB_DB") or "LinkProDB"
	coll_name = os.getenv("MONGODB_COLLECTION") or "linkprosessions"
	
	client = MongoClient(uri, serverSelectionTimeoutMS=5000)
	collection = client[db_name][coll_name]
	
	print("✅ Connected to MongoDB")
	
	azure_prefix = "https://linkproaccstorageadmin.blob.core.windows.net"
	
	# Base filter criteria (same for both ranges)
	base_filter = {
		"videoUrl": {"$exists": True, "$regex": f"^{azure_prefix.replace('.', '\\.')}"},
		"videoDuration": {"$gte": 90},
		"supportIdMember": {"$ne": None},
	}
	
	# Range 1: March 1st, 2025 onwards (232 sessions - already processed)
	march_2025_start = parse_iso("2025-03-01T00:00:00.000Z")
	range1_filter = {**base_filter, "createdAt": {"$gte": march_2025_start}}
	
	# Range 2: Oct 1st, 2024 to Mar 1st, 2025 (559 sessions - about to process)
	oct_2024_start = parse_iso("2024-10-01T00:00:00.000Z")
	march_2025_end = parse_iso("2025-03-01T00:00:00.000Z")
	range2_filter = {**base_filter, "createdAt": {"$gte": oct_2024_start, "$lt": march_2025_end}}
	
	print("\n📊 Calculating durations...")
	
	# Calculate Range 1 (already processed)
	range1_pipeline = [
		{"$match": range1_filter},
		{"$group": {
			"_id": None,
			"count": {"$sum": 1},
			"total_duration": {"$sum": "$videoDuration"}
		}}
	]
	
	range1_result = list(collection.aggregate(range1_pipeline))
	range1_count = range1_result[0]["count"] if range1_result else 0
	range1_duration = range1_result[0]["total_duration"] if range1_result else 0
	
	# Calculate Range 2 (about to process)
	range2_pipeline = [
		{"$match": range2_filter},
		{"$group": {
			"_id": None,
			"count": {"$sum": 1},
			"total_duration": {"$sum": "$videoDuration"}
		}}
	]
	
	range2_result = list(collection.aggregate(range2_pipeline))
	range2_count = range2_result[0]["count"] if range2_result else 0
	range2_duration = range2_result[0]["total_duration"] if range2_result else 0
	
	# Calculate totals
	total_count = range1_count + range2_count
	total_duration = range1_duration + range2_duration
	
	print(f"\n🎬 SESSION DURATION SUMMARY")
	print(f"=" * 50)
	print(f"📅 Range 1 (Mar 1st, 2025+): {range1_count} sessions")
	print(f"   Duration: {format_duration(range1_duration)} ({range1_duration:.0f} seconds)")
	print()
	print(f"📅 Range 2 (Oct 1st, 2024 - Mar 1st, 2025): {range2_count} sessions")
	print(f"   Duration: {format_duration(range2_duration)} ({range2_duration:.0f} seconds)")
	print()
	print(f"🎯 TOTAL: {total_count} sessions")
	print(f"🕐 TOTAL DURATION: {format_duration(total_duration)} ({total_duration:.0f} seconds)")
	print(f"=" * 50)
	
	# Additional stats
	avg_duration = total_duration / total_count if total_count > 0 else 0
	print(f"📊 Average session duration: {format_duration(avg_duration)}")
	
	client.close()

if __name__ == "__main__":
	main()
