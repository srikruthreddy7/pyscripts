#!/usr/bin/env python3
import os, sys, re, json
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pymongo.monitoring import CommandListener
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*_a, **_k): return False
try:
    from bson.json_util import dumps
except Exception:
    dumps = None

class FindLogger(CommandListener):
    def started(self, event):
        if event.command_name == 'find':
            print('REQUEST:')
            if dumps: print(dumps(event.command, indent=2, sort_keys=True))
            else: print(event.command)
    def succeeded(self, event):
        if event.command_name == 'find':
            print('RESPONSE:')
            if dumps: print(dumps(event.reply, indent=2, sort_keys=True))
            else: print(event.reply)
    def failed(self, event):
        if event.command_name == 'find':
            print(f'RESPONSE ERROR: {event.failure}', file=sys.stderr)

def main():
    load_dotenv()
    uri = os.getenv('MONGODB_URI') or os.getenv('MONGODB_CONNECTION_STRING') or os.getenv('MONGO_URI')
    db_name = os.getenv('MONGODB_DB_NAME') or os.getenv('MONGODB_DB') or 'LinkProDB'
    coll_name = os.getenv('MONGODB_COLLECTION') or 'linkprosessions'
    if not uri: print('Missing MONGODB_URI', file=sys.stderr); sys.exit(2)
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000, event_listeners=[FindLogger()])
        client.admin.command('ping')
    except ConnectionFailure as e:
        print(f'Connection failed: {e}', file=sys.stderr); sys.exit(1)
    except Exception as e:
        print(f'Error connecting: {e}', file=sys.stderr); sys.exit(1)
    database = client[db_name]
    collection = database[coll_name]

    # Flexible env-driven query controls
    raw_filter = os.getenv('QUERY_FILTER') or '{}'
    raw_fields = os.getenv('QUERY_FIELDS')  # projection
    raw_sort = os.getenv('QUERY_SORT')      # e.g. {"createdAt": -1}
    limit = int(os.getenv('QUERY_LIMIT') or '5')
    print_count = (os.getenv('PRINT_COUNT') or '0') not in ('0','false','False')

    try:
        filt = json.loads(raw_filter)
        if not isinstance(filt, dict): raise ValueError('QUERY_FILTER must be a JSON object')
    except Exception as e:
        print(f'Invalid QUERY_FILTER: {e}', file=sys.stderr); sys.exit(2)

    proj = None
    if raw_fields:
        try:
            proj = json.loads(raw_fields)
            if not isinstance(proj, dict): raise ValueError('QUERY_FIELDS must be a JSON object')
        except Exception as e:
            print(f'Invalid QUERY_FIELDS: {e}', file=sys.stderr); sys.exit(2)

    sort_spec = None
    if raw_sort:
        try:
            sort_obj = json.loads(raw_sort)
            if isinstance(sort_obj, dict):
                sort_spec = list(sort_obj.items())
            elif isinstance(sort_obj, list):
                sort_spec = sort_obj
            else:
                raise ValueError('QUERY_SORT must be an object or array')
        except Exception as e:
            print(f'Invalid QUERY_SORT: {e}', file=sys.stderr); sys.exit(2)

    # Convenience toggles
    since_iso = os.getenv('SINCE_ISO')
    if since_iso:
        try:
            since = datetime.fromisoformat(since_iso.replace('Z','+00:00'))
            if since.tzinfo is None: since = since.replace(tzinfo=timezone.utc)
            created = filt.get('createdAt') or {}
            created['$gte'] = since
            filt['createdAt'] = created
        except Exception as e:
            print(f'Invalid SINCE_ISO: {e}', file=sys.stderr); sys.exit(2)

    video_prefix = os.getenv('VIDEO_PREFIX')
    if video_prefix:
        video = filt.get('videoUrl') or {}
        video['$regex'] = '^' + re.escape(video_prefix)
        filt['videoUrl'] = video

    if (os.getenv('SUPPORT_NOT_NULL') or '0') not in ('0','false','False'):
        filt['supportIdMember'] = {'$ne': None}

    # Show what we're about to run
    print('EFFECTIVE FILTER:')
    if dumps: print(dumps(filt, indent=2, sort_keys=True))
    else: print(filt)
    if proj is not None:
        print('PROJECTION:')
        if dumps: print(dumps(proj, indent=2, sort_keys=True))
        else: print(proj)
    if sort_spec is not None:
        print(f'SORT: {sort_spec}')
    print(f'LIMIT: {limit}')

    try:
        docs = []
        if limit and limit > 0:
            cursor = collection.find(filt, proj) if proj is not None else collection.find(filt)
            if sort_spec: cursor = cursor.sort(sort_spec)
            docs = list(cursor.limit(limit))
    except Exception as e:
        print(f'Query error: {e}', file=sys.stderr); sys.exit(1)

    if print_count:
        try:
            cnt = collection.count_documents(filt)
            print(f'COUNT: {cnt}')
        except Exception as e:
            print(f'Count error: {e}', file=sys.stderr)

    if not docs:
        print('No documents.')
    print_docs_flag = (os.getenv('PRINT_DOCS') or '1') not in ('0','false','False')
    if print_docs_flag and docs:
        print(f'Documents (up to {len(docs)}) from {db_name}.{coll_name}:')
        for d in docs:
            if dumps: print(dumps(d, indent=2, sort_keys=True))
            else: print(d)

    # Monthly aggregation visualization
    if (os.getenv('AGG_MONTHLY') or '0') not in ('0','false','False'):
        try:
            pipeline = [
                { '$match': filt },
                { '$group': { '_id': { '$dateToString': { 'format': '%Y-%m', 'date': '$createdAt' } }, 'count': { '$sum': 1 } } },
                { '$sort': { '_id': 1 } }
            ]
            buckets = list(collection.aggregate(pipeline))
            print('Monthly counts:')
            for b in buckets:
                ym = b.get('_id')
                c = b.get('count')
                print(f'{ym}: {c}')
        except Exception as e:
            print(f'Monthly aggregation error: {e}', file=sys.stderr)

if __name__ == '__main__':
    main()


