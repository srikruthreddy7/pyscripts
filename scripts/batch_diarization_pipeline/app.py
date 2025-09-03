"""
Batch Audio Transcription and Diarization Pipeline on Modal

This pipeline processes ~467 hours of 16kHz FLAC audio files to produce
diarized transcripts with speaker labels using open-source models:
- ASR: nvidia/parakeet-tdt-0.6b-v2 
- Diarization: nvidia/diar_streaming_sortformer_4spk-v2

Cost target: ≤ $10 total for the entire job via smart batching and throughput optimization.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import modal

from .config import (
    ASR_MODEL_ID,
    DIARIZATION_MODEL_ID,
    GPU_TYPE,
    DEFAULT_GPU_BATCH_SIZE,
    DEFAULT_NUM_REQUESTS,
    DIARIZATION_PRESETS,
    get_modal_image,
    AUDIO_VOLUME_NAME,
    MODELS_CACHE_VOLUME_NAME,
    RESULTS_VOLUME_NAME
)
from .indexer import AudioIndexer, RequestPartitioner
from .utils import setup_logging, generate_cost_report

# Modal app setup
app = modal.App("batch-diarization-pipeline")

# Volume mounts
# Uses specific names with env-var overrides in config.py
audio_volume = modal.Volume.from_name(AUDIO_VOLUME_NAME, create_if_missing=True)
models_cache_volume = modal.Volume.from_name(MODELS_CACHE_VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

# Image with all dependencies
CODE_DIR = Path(__file__).parent
# Image now includes the package via get_modal_image(); no extra add_local_dir here.
image = get_modal_image()

@app.function(
    image=image,
    volumes={
        "/vol/audio": audio_volume,
        "/vol/models-cache": models_cache_volume,
        "/vol/results": results_volume,
    },
    timeout=3600,  # 1 hour timeout for indexing
)
def index_audio_files(
    job_id: str,
    audio_subdir: Optional[str] = None,
    limit_files: Optional[int] = None,
    debug: bool = False,
) -> Dict:
    """
    Index all FLAC audio files in the volume and create request partitions.
    
    Args:
        job_id: Unique identifier for this job
        
    Returns:
        Dict with indexing stats and partition info
    """
    # Elevate logging to DEBUG for deep inspection when requested
    if debug:
        from .utils import setup_logging as _setup
        _setup(level="DEBUG")
    else:
        setup_logging()
    
    # Allow targeting a subdirectory inside the audio volume (e.g., 2025/08/28)
    audio_root = "/vol/audio" if not audio_subdir else str(Path("/vol/audio") / audio_subdir)
    indexer = AudioIndexer(audio_root)
    
    # Scan and index all audio files
    print(f"🔍 Scanning audio files in: {audio_root}")
    index_data = indexer.scan_audio_files(limit=limit_files)
    
    # Save index to results volume
    index_path = f"/vol/results/index_{job_id}.jsonl"
    indexer.save_index(index_data, index_path)
    
    print(f"📊 Indexed {len(index_data)} files")
    print(f"📊 Total duration: {sum(f['duration_sec'] for f in index_data):.1f} seconds")
    if debug:
        # Print a concise per-file summary for quick inspection
        for f in index_data[:10]:
            print(
                f" - {f['relpath']}: dur={f['duration_sec']:.3f}s, sr={f['sample_rate']}, ch={f['channels']}, bytes={f['bytes']}"
            )
    print(f"📊 Total size: {sum(f['bytes'] for f in index_data) / (1024**3):.1f} GB")
    
    return {
        "index_path": index_path,
        "file_count": len(index_data),
        "total_duration_sec": sum(f['duration_sec'] for f in index_data),
        "total_bytes": sum(f['bytes'] for f in index_data)
    }


@app.function(
    image=image,
    volumes={
        "/vol/audio": audio_volume,
        "/vol/models-cache": models_cache_volume,
        "/vol/results": results_volume,
    },
    timeout=1800,
)
def debug_index_sample(job_id: str, audio_subdir: Optional[str] = None, n: int = 10) -> Dict:
    """Index only the first N audio files with verbose debug logs.

    Returns the same dict as index_audio_files but limited to N files.
    """
    return index_audio_files.local(job_id=job_id, audio_subdir=audio_subdir, limit_files=n, debug=True)

@app.function(
    image=image,
    volumes={
        "/vol/audio": audio_volume,
        "/vol/models-cache": models_cache_volume,
        "/vol/results": results_volume,
    },
    timeout=1800,  # 30 minutes for partitioning
)
def partition_requests(
    index_path: str,
    num_requests: int = DEFAULT_NUM_REQUESTS
) -> List[Dict]:
    """
    Partition the indexed files into balanced request batches.
    
    Args:
        index_path: Path to the index file
        num_requests: Number of parallel requests to create
        
    Returns:
        List of request metadata with file assignments
    """
    setup_logging()
    
    # Load index data
    with open(index_path, 'r') as f:
        index_data = [json.loads(line) for line in f]
    
    partitioner = RequestPartitioner(num_requests)
    
    print(f"📦 Partitioning {len(index_data)} files into {num_requests} requests...")
    requests = partitioner.partition_files(index_data)
    
    # Save request manifests
    job_id = Path(index_path).stem.replace("index_", "")
    requests_path = f"/vol/results/requests_{job_id}.jsonl"
    
    with open(requests_path, 'w') as f:
        for i, request in enumerate(requests):
            request_data = {
                "request_id": i,
                "file_count": len(request["files"]),
                "total_duration_sec": sum(f["duration_sec"] for f in request["files"]),
                "total_bytes": sum(f["bytes"] for f in request["files"]),
                "files": request["files"]
            }
            f.write(json.dumps(request_data) + "\n")
    
    print(f"✅ Created {len(requests)} balanced request partitions")
    for i, req in enumerate(requests):
        print(f"  Request {i}: {len(req['files'])} files, "
              f"{sum(f['duration_sec'] for f in req['files']):.1f}s, "
              f"{sum(f['bytes'] for f in req['files']) / (1024**2):.1f} MB")
    
    return [{"request_id": i, "requests_path": requests_path} for i in range(len(requests))]

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        "/vol/audio": audio_volume,
        "/vol/models-cache": models_cache_volume,
        "/vol/results": results_volume,
    },
    timeout=7200,  # 2 hours per request
    scaledown_window=300,
)
def process_transcription_request(
    request_id: int,
    requests_path: str,
    job_id: str,
    gpu_batch_size: int = DEFAULT_GPU_BATCH_SIZE,
    model_id: str = ASR_MODEL_ID
) -> Dict:
    """
    Process a single transcription request with batched ASR.
    
    Args:
        request_id: ID of the request to process
        requests_path: Path to the requests manifest file
        job_id: Job identifier
        gpu_batch_size: Batch size for GPU processing
        model_id: ASR model to use
        
    Returns:
        Dict with processing stats
    """
    setup_logging()
    start_time = time.time()
    
    # Load request data
    with open(requests_path, 'r') as f:
        requests = [json.loads(line) for line in f]
    
    request_data = requests[request_id]
    files = request_data["files"]
    
    print(f"🎤 Processing transcription request {request_id} with {len(files)} files")
    
    # Initialize ASR service (import inside container to avoid local heavy deps)
    from .asr_service import ASRService
    asr_service = ASRService(model_id, gpu_batch_size, "/vol/models-cache")
    
    # Process files
    results = asr_service.process_files(files, f"/vol/results/asr_interim_{job_id}", request_id)
    
    processing_time = time.time() - start_time
    
    print(f"✅ Completed transcription request {request_id} in {processing_time:.1f}s")
    
    return {
        "request_id": request_id,
        "files_processed": len(results),
        "processing_time_sec": processing_time,
        "gpu_hours": processing_time / 3600,
        "results_path": f"/vol/results/asr_interim_{job_id}/request_{request_id}.jsonl"
    }

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        "/vol/audio": audio_volume,
        "/vol/models-cache": models_cache_volume,
        "/vol/results": results_volume,
    },
    timeout=7200,  # 2 hours per request
    scaledown_window=300,
)
def process_diarization_request(
    request_id: int,
    requests_path: str,
    job_id: str,
    preset: str = "high_latency",
    model_id: str = DIARIZATION_MODEL_ID
) -> Dict:
    """
    Process a single diarization request and fuse with ASR results.
    
    Args:
        request_id: ID of the request to process
        requests_path: Path to the requests manifest file
        job_id: Job identifier
        preset: Diarization preset to use (high_latency, very_high_latency)
        model_id: Diarization model to use
        
    Returns:
        Dict with processing stats
    """
    setup_logging()
    start_time = time.time()
    
    # Load request data
    with open(requests_path, 'r') as f:
        requests = [json.loads(line) for line in f]
    
    request_data = requests[request_id]
    files = request_data["files"]
    
    print(f"🗣️ Processing diarization request {request_id} with {len(files)} files")
    
    # Initialize services (import inside container)
    from .diarization_service import DiarizationService
    from .fusion_service import FusionService
    diarization_service = DiarizationService(model_id, preset, "/vol/models-cache")
    fusion_service = FusionService()
    
    # Load ASR results
    asr_results_path = f"/vol/results/asr_interim_{job_id}/request_{request_id}.jsonl"
    with open(asr_results_path, 'r') as f:
        # Map ASR results by basename for robust lookup
        parsed = [json.loads(line) for line in f]
        asr_results = {Path(item["filename"]).name: item for item in parsed}
    
    # Process diarization and fusion for each file
    final_results = []
    for file_info in files:
        # Prefer absolute path saved during indexing
        file_path = file_info.get("abspath") or str(Path("/vol/audio") / file_info["relpath"]) 

        # Get ASR result
        asr_key = Path(file_info["relpath"]).name
        if asr_key not in asr_results:
            raise KeyError(f"ASR result not found for {asr_key} in {asr_results_path}")
        asr_result = asr_results[asr_key]
        
        # Run diarization
        diarization_result = diarization_service.process_file(file_path, file_info)
        
        # Fuse ASR and diarization
        fused_result = fusion_service.fuse_results(asr_result, diarization_result, file_info)
        final_results.append(fused_result)
    
    # Save final results
    output_dir = f"/vol/results/final_{job_id}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"/vol/results/srt_{job_id}", exist_ok=True)
    
    for result in final_results:
        # Save JSONL
        output_path = f"{output_dir}/{Path(result['filename']).stem}.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Save SRT (optional)
        srt_path = f"/vol/results/srt_{job_id}/{Path(result['filename']).stem}.srt"
        fusion_service.save_srt(result, srt_path)
    
    processing_time = time.time() - start_time
    
    print(f"✅ Completed diarization request {request_id} in {processing_time:.1f}s")
    
    return {
        "request_id": request_id,
        "files_processed": len(final_results),
        "processing_time_sec": processing_time,
        "gpu_hours": processing_time / 3600,
        "preset_used": preset
    }

@app.local_entrypoint()
def batch_transcription(
    job_id: str,
    model_id: str = ASR_MODEL_ID,
    gpu_type: str = GPU_TYPE,
    gpu_batch_size: int = DEFAULT_GPU_BATCH_SIZE,
    num_requests: int = DEFAULT_NUM_REQUESTS,
    audio_subdir: Optional[str] = None
):
    """
    Run the complete batch transcription pipeline.
    
    Example usage:
    modal run app.py::batch_transcription --job-id hvac-2025-01 --num-requests 25
    """
    print(f"🚀 Starting batch transcription job: {job_id}")
    print(f"📊 Configuration:")
    print(f"  Model: {model_id}")
    print(f"  GPU: {gpu_type}")
    print(f"  Batch size: {gpu_batch_size}")
    print(f"  Concurrent requests: {num_requests}")
    
    # Step 1: Index audio files
    print("\n📂 Step 1: Indexing audio files...")
    index_result = index_audio_files.remote(job_id, audio_subdir)
    
    # Step 2: Partition into requests  
    print("\n📦 Step 2: Partitioning requests...")
    partition_results = partition_requests.remote(
        index_result["index_path"], 
        num_requests
    )
    
    # Step 3: Process transcription requests in parallel
    print(f"\n🎤 Step 3: Processing {num_requests} transcription requests...")
    requests_path = partition_results[0]["requests_path"]
    
    transcription_futures = []
    for i in range(num_requests):
        future = process_transcription_request.spawn(
            i, requests_path, job_id, gpu_batch_size, model_id
        )
        transcription_futures.append(future)
    
    # Wait for all transcription requests to complete
    transcription_results = [f.get() for f in transcription_futures]
    
    # Generate cost report
    total_gpu_hours = sum(r["gpu_hours"] for r in transcription_results)
    
    print(f"\n✅ Transcription complete!")
    print(f"📊 Total GPU hours: {total_gpu_hours:.2f}")
    print(f"💰 Estimated cost: ${total_gpu_hours * 1.95:.2f}")
    print(f"📁 ASR results saved to: /vol/results/asr_interim_{job_id}/")

@app.local_entrypoint()
def test_small_transcription(
    job_id: str = "test-small",
    num_files: int = 5
):
    """
    Test transcription on a small number of files to validate it works.
    """
    print(f"🧪 Testing transcription on {num_files} files from job: {job_id}")
    
    # Use existing request partitions but only process first few files
    requests_path = f"/vol/results/requests_dbg-001.jsonl"
    
    # Load just the first request and limit files
    with open(requests_path, 'r') as f:
        requests = [json.loads(line) for line in f]
    
    first_request = requests[0]  
    limited_files = first_request["files"][:num_files]  # Take only first N files
    
    print(f"🎤 Testing with {len(limited_files)} files (batch size: 2)")
    
    # Process with very small batch size
    future = process_transcription_request.spawn(
        0, requests_path, "test-small", 2, ASR_MODEL_ID  # batch_size=2
    )
    
    result = future.get()
    print(f"✅ Test completed!")
    print(f"📊 Files processed: {result['files_processed']}")
    print(f"💰 Cost: ${result['gpu_hours'] * 1.95:.3f}")
    
    return result

@app.local_entrypoint()
def transcribe_from_existing_requests(
    job_id: str,
    model_id: str = ASR_MODEL_ID,
    gpu_batch_size: int = DEFAULT_GPU_BATCH_SIZE,
    num_requests: int = DEFAULT_NUM_REQUESTS
):
    """
    Run transcription using existing request partitions (skip indexing/partitioning).
    
    Example usage:
    modal run app.py::transcribe_from_existing_requests --job-id dbg-001 --num-requests 10
    """
    print(f"🚀 Starting transcription job using existing requests: {job_id}")
    print(f"📊 Configuration:")
    print(f"  Model: {model_id}")
    print(f"  GPU: {GPU_TYPE}")
    print(f"  Batch size: {gpu_batch_size}")
    print(f"  Concurrent requests: {num_requests}")
    
    # Use existing request partitions
    requests_path = f"/vol/results/requests_{job_id}.jsonl"
    
    # Step 3: Process transcription requests in parallel
    print(f"\n🎤 Processing {num_requests} transcription requests...")
    
    transcription_futures = []
    for i in range(num_requests):
        future = process_transcription_request.spawn(
            i, requests_path, job_id, gpu_batch_size, model_id
        )
        transcription_futures.append(future)
    
    # Wait for all transcription requests to complete
    transcription_results = [f.get() for f in transcription_futures]
    
    # Generate cost report
    total_gpu_hours = sum(r["gpu_hours"] for r in transcription_results)
    
    print(f"\n✅ Transcription complete!")
    print(f"📊 Total GPU hours: {total_gpu_hours:.2f}")
    print(f"💰 Estimated cost: ${total_gpu_hours * 1.95:.2f}")
    print(f"📁 ASR results saved to: /vol/results/asr_interim_{job_id}/")

@app.local_entrypoint()
def batch_diarize_and_fuse(
    job_id: str,
    preset: str = "high_latency",
    model_id: str = DIARIZATION_MODEL_ID,
    num_requests: int = DEFAULT_NUM_REQUESTS
):
    """
    Run diarization and fusion on existing ASR results.
    
    Example usage:
    modal run app.py::batch_diarize_and_fuse --job-id hvac-2025-01 --preset high_latency
    """
    print(f"🚀 Starting batch diarization and fusion job: {job_id}")
    print(f"📊 Configuration:")
    print(f"  Diarization model: {model_id}")
    print(f"  Preset: {preset}")
    print(f"  RTF: {DIARIZATION_PRESETS[preset]['rtf']}")
    print(f"  Concurrent requests: {num_requests}")
    
    # Load existing request partitions
    requests_path = f"/vol/results/requests_{job_id}.jsonl"
    
    # Process diarization requests in parallel
    print(f"\n🗣️ Processing {num_requests} diarization requests...")
    
    diarization_futures = []
    for i in range(num_requests):
        future = process_diarization_request.spawn(
            i, requests_path, job_id, preset, model_id
        )
        diarization_futures.append(future)
    
    # Wait for all diarization requests to complete
    diarization_results = [f.get() for f in diarization_futures]
    
    # Generate final cost report
    total_gpu_hours = sum(r["gpu_hours"] for r in diarization_results)
    cost_report = generate_cost_report(job_id, diarization_results, preset)
    
    print(f"\n✅ Diarization and fusion complete!")
    print(f"📊 Total GPU hours: {total_gpu_hours:.2f}")
    print(f"💰 Estimated cost: ${total_gpu_hours * 1.95:.2f}")
    print(f"📁 Final results saved to: /vol/results/final_{job_id}/")
    print(f"📁 SRT files saved to: /vol/results/srt_{job_id}/")
    print(f"📊 Cost report: {cost_report}")

@app.local_entrypoint()
def full_pipeline(
    job_id: str,
    preset: str = "high_latency",
    gpu_batch_size: int = DEFAULT_GPU_BATCH_SIZE,
    num_requests: int = DEFAULT_NUM_REQUESTS,
    audio_subdir: Optional[str] = None
):
    """
    Run the complete pipeline: transcription + diarization + fusion.
    
    Example usage:
    modal run app.py::full_pipeline --job-id hvac-2025-01 --preset high_latency --num-requests 25
    """
    print(f"🚀 Starting full pipeline job: {job_id}")
    
    # Run transcription
    batch_transcription(job_id, ASR_MODEL_ID, GPU_TYPE, gpu_batch_size, num_requests, audio_subdir)
    
    # Run diarization and fusion  
    batch_diarize_and_fuse(job_id, preset, DIARIZATION_MODEL_ID, num_requests)
    
    print(f"\n🎉 Full pipeline complete for job: {job_id}")

@app.local_entrypoint()
def verify_inputs(audio_subdir: Optional[str] = None):
    """
    Quickly verify input volume contents and size.

    Example:
      modal run app.py::verify_inputs --audio-subdir 2025/08/28
    """
    target = str(Path("/vol/audio") / audio_subdir) if audio_subdir else "/vol/audio"
    print(f"🔎 Verifying inputs under: {target}")

    # Reuse indexer logic to count files and sizes without saving index
    idx = AudioIndexer(target)
    data = idx.scan_audio_files()
    total_bytes = sum(f["bytes"] for f in data)
    total_files = len(data)
    total_gib = total_bytes / (1024**3)
    print(f"📁 Files: {total_files}")
    print(f"💾 Size: {total_gib:.1f} GiB")
    if total_files == 0:
        print("⚠️ No supported audio files found. Ensure .flac files exist and path is correct.")
    return {"files": total_files, "size_gib": round(total_gib, 2)}

@app.local_entrypoint()
def estimate_runtime_and_cost(
    job_id: str,
    preset: str = "high_latency",
    num_requests: int = DEFAULT_NUM_REQUESTS,
    audio_subdir: Optional[str] = None
):
    """
    Compute exact runtime and cost based on indexed audio duration.

    Runs indexing on the specified subdir, then computes:
    - Total audio hours
    - Estimated GPU hours (ASR + diarization)
    - Estimated cost (with overhead)
    - Estimated wall-clock time with parallelism

    Example:
      modal run app.py::estimate_runtime_and_cost \
        --job-id hvac-2025-08-28 --preset high_latency \
        --num-requests 25 --audio-subdir 2025/08/28
    """
    from .config import estimate_cost, DIARIZATION_PRESETS

    print("🧮 Estimating runtime and cost...")
    idx = index_audio_files.remote(job_id, audio_subdir)

    total_sec = float(idx.get("total_duration_sec", 0.0))
    total_bytes = float(idx.get("total_bytes", 0))
    file_count = int(idx.get("file_count", 0))

    total_hours = total_sec / 3600.0
    total_gib = total_bytes / (1024**3)

    # RTF assumptions: ASR ~0.005, diarization per preset
    asr_rtf = 0.005
    diar_rtf = float(DIARIZATION_PRESETS[preset]["rtf"])
    sequential_gpu_hours = total_hours * (asr_rtf + diar_rtf)
    parallel_gpu_hours = sequential_gpu_hours / max(1, num_requests)
    overhead_factor = 1.2
    est_wall_hours = parallel_gpu_hours * overhead_factor

    # Cost using config's estimator (includes overhead)
    cost = estimate_cost(total_hours, preset)

    print("\n===== INPUT DATA =====")
    print(f"Files: {file_count}")
    print(f"Size: {total_gib:.2f} GiB")
    print(f"Total audio: {total_hours:.2f} hours")

    print("\n===== CONFIG =====")
    print(f"Preset: {preset} (diar RTF={diar_rtf})")
    print(f"ASR RTF: {asr_rtf}")
    print(f"Parallel requests: {num_requests}")

    print("\n===== ESTIMATES =====")
    print(f"GPU hours (sequential, ASR+diar): {sequential_gpu_hours:.3f}")
    print(f"Estimated wall time (hours): {est_wall_hours:.2f}")
    print(f"Estimated wall time (hh:mm): {int(est_wall_hours):02d}:{int((est_wall_hours%1)*60):02d}")

    print("\n===== COST =====")
    print(f"GPU hours (diar): {cost['diarization']['gpu_hours']:.3f}")
    print(f"GPU hours (asr): {cost['asr']['gpu_hours']:.3f}")
    print(f"Cost before overhead: ${cost['total']['cost_usd_before_overhead']:.2f}")
    print(f"Total cost with overhead ({cost['total']['overhead_factor']}x): ${cost['total']['cost_usd_with_overhead']:.2f}")

    return {
        "files": file_count,
        "size_gib": round(total_gib, 2),
        "total_hours": round(total_hours, 3),
        "gpu_hours_sequential": round(sequential_gpu_hours, 3),
        "estimated_wall_hours": round(est_wall_hours, 2),
        "cost": cost,
    }

if __name__ == "__main__":
    # For local testing
    pass
