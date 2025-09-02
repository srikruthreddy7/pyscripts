"""
Audio file indexing and request partitioning for batch processing.

This module handles:
1. Scanning and indexing audio files from the volume
2. Validating audio format and metadata
3. Partitioning files into balanced request batches for parallel processing
"""

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import logging

from .config import (
    REQUIRED_SAMPLE_RATE, 
    REQUIRED_CHANNELS, 
    SUPPORTED_FORMATS,
    MAX_FILE_SIZE_GB,
    MIN_AUDIO_DURATION_SEC,
    MAX_AUDIO_DURATION_SEC
)

logger = logging.getLogger(__name__)

class AudioIndexer:
    """Handles scanning and indexing of audio files."""
    
    def __init__(self, audio_root: str):
        """
        Initialize the audio indexer.
        
        Args:
            audio_root: Root directory containing audio files
        """
        self.audio_root = Path(audio_root)
        
    def get_audio_metadata(self, file_path: str) -> Dict:
        """
        Extract audio metadata using ffprobe (fast, no decoding).
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dict with duration, sample_rate, channels, and validation info
        """
        try:
            # Use ffprobe to get audio metadata without decoding
            cmd = [
                "ffprobe", 
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.warning(f"ffprobe failed for {file_path}: {result.stderr}")
                return None
                
            metadata = json.loads(result.stdout)
            
            # Find the audio stream
            audio_stream = None
            for stream in metadata.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_stream = stream
                    break
                    
            if not audio_stream:
                logger.warning(f"No audio stream found in {file_path}")
                return None
                
            # Extract relevant information with robust duration fallback
            def _parse_float(x):
                try:
                    return float(x)
                except Exception:
                    return 0.0

            sample_rate = int(audio_stream.get("sample_rate", 0) or 0)
            channels = int(audio_stream.get("channels", 0) or 0)

            # 1) Prefer stream duration when present and valid
            duration = _parse_float(audio_stream.get("duration")) if audio_stream.get("duration") not in (None, "N/A") else 0.0

            # 2) Fallback to container/format duration
            if duration <= 0:
                fmt = metadata.get("format", {})
                fmt_duration = fmt.get("duration")
                if fmt_duration not in (None, "N/A"):
                    duration = _parse_float(fmt_duration)

            # 3) Fallback to nb_samples/sample_rate if available
            if duration <= 0:
                nb_samples = audio_stream.get("nb_samples")
                if nb_samples is not None and sample_rate:
                    try:
                        duration = float(nb_samples) / float(sample_rate)
                    except Exception:
                        pass

            # 4) Fallback to duration_ts * time_base
            if duration <= 0:
                dur_ts = audio_stream.get("duration_ts")
                time_base = audio_stream.get("time_base")  # e.g., "1/16000"
                if dur_ts and time_base and isinstance(time_base, str) and "/" in time_base:
                    try:
                        num, den = time_base.split("/")
                        tb = float(num) / float(den)
                        duration = float(dur_ts) * tb
                    except Exception:
                        pass

            # 5) External tool fallback: soxi -D (fast metadata read)
            if duration <= 0:
                try:
                    soxi = subprocess.run(
                        ["soxi", "-D", file_path], capture_output=True, text=True, timeout=15
                    )
                    if soxi.returncode == 0:
                        d = soxi.stdout.strip()
                        if d and d.lower() != "n/a":
                            duration = float(d)
                except Exception:
                    pass

            # 6) Library fallback: soundfile.info (no full decode)
            if duration <= 0:
                try:
                    import soundfile as sf  # type: ignore
                    info = sf.info(file_path)
                    if info.samplerate and info.frames:
                        duration = float(info.frames) / float(info.samplerate)
                        # Fill missing basic fields from soundfile if needed
                        if sample_rate == 0:
                            sample_rate = int(info.samplerate)
                        if channels == 0:
                            channels = int(info.channels)
                except Exception:
                    pass
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            return {
                "duration_sec": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "bytes": file_size,
                "codec": audio_stream.get("codec_name"),
                "bit_rate": audio_stream.get("bit_rate"),
                "valid": self._validate_audio_specs(duration, sample_rate, channels, file_size)
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"ffprobe timeout for {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error getting metadata for {file_path}: {e}")
            return None
            
    def _validate_audio_specs(self, duration: float, sample_rate: int, channels: int, file_size: int) -> bool:
        """
        Validate audio file specifications against requirements.
        
        Args:
            duration: Audio duration in seconds
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            file_size: File size in bytes
            
        Returns:
            True if file meets requirements
        """
        issues = []
        
        # Check sample rate
        if sample_rate != REQUIRED_SAMPLE_RATE:
            issues.append(f"Sample rate {sample_rate}Hz != required {REQUIRED_SAMPLE_RATE}Hz")
            
        # Check channels
        if channels != REQUIRED_CHANNELS:
            issues.append(f"Channels {channels} != required {REQUIRED_CHANNELS}")
            
        # Check duration
        if duration < MIN_AUDIO_DURATION_SEC:
            issues.append(f"Duration {duration}s < minimum {MIN_AUDIO_DURATION_SEC}s")
        elif duration > MAX_AUDIO_DURATION_SEC:
            issues.append(f"Duration {duration}s > maximum {MAX_AUDIO_DURATION_SEC}s")
            
        # Check file size
        file_size_gb = file_size / (1024**3)
        if file_size_gb > MAX_FILE_SIZE_GB:
            issues.append(f"File size {file_size_gb:.1f}GB > maximum {MAX_FILE_SIZE_GB}GB")
            
        if issues:
            logger.warning(f"Validation issues: {'; '.join(issues)}")
            return False
            
        return True
        
    def scan_audio_files(self) -> List[Dict]:
        """
        Scan the audio directory and index all supported audio files.
        
        Returns:
            List of file metadata dictionaries
        """
        logger.info(f"Scanning audio files in {self.audio_root}")
        
        audio_files = []
        total_files = 0
        valid_files = 0
        
        # Walk through all files in the audio directory
        for root, dirs, files in os.walk(self.audio_root):
            for file in files:
                total_files += 1
                
                # Check file extension
                file_path = Path(root) / file
                if file_path.suffix.lower() not in SUPPORTED_FORMATS:
                    continue
                    
                # Get relative path from audio root
                rel_path = file_path.relative_to(self.audio_root)
                
                # Get metadata
                metadata = self.get_audio_metadata(str(file_path))
                if metadata is None:
                    continue
                    
                # Add file info
                file_info = {
                    "relpath": str(rel_path),
                    "abspath": str(file_path),
                    **metadata
                }
                
                audio_files.append(file_info)
                
                if metadata["valid"]:
                    valid_files += 1
                    
                # Log progress every 100 files
                if len(audio_files) % 100 == 0:
                    logger.info(f"Processed {len(audio_files)} audio files...")
                    
        logger.info(f"Scanning complete: {total_files} total files, "
                   f"{len(audio_files)} audio files, {valid_files} valid files")
        
        # Sort by path for consistent ordering
        audio_files.sort(key=lambda x: x["relpath"])
        
        return audio_files
        
    def save_index(self, index_data: List[Dict], output_path: str):
        """
        Save the index data to a JSONL file.
        
        Args:
            index_data: List of file metadata
            output_path: Path to save the index file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for item in index_data:
                f.write(json.dumps(item) + '\n')
                
        logger.info(f"Saved index with {len(index_data)} files to {output_path}")


class RequestPartitioner:
    """Handles partitioning of files into balanced request batches."""
    
    def __init__(self, num_requests: int):
        """
        Initialize the request partitioner.
        
        Args:
            num_requests: Number of parallel requests to create
        """
        self.num_requests = num_requests
        
    def partition_files(self, files: List[Dict]) -> List[Dict]:
        """
        Partition files into balanced request batches.
        
        Uses the Modal blog approach:
        1. Shuffle files to break ordering correlations
        2. Round-robin deal into num_requests piles
        3. Balance by total bytes, file count, and duration
        4. Optional greedy rebalancing
        
        Args:
            files: List of file metadata dictionaries
            
        Returns:
            List of request batches, each containing file assignments
        """
        logger.info(f"Partitioning {len(files)} files into {self.num_requests} requests")
        
        # Filter to only valid files
        valid_files = [f for f in files if f.get("valid", False)]
        logger.info(f"Using {len(valid_files)} valid files for partitioning")
        
        if len(valid_files) == 0:
            logger.error("No valid files to partition")
            return []
            
        # Step 1: Shuffle to break ordering correlations
        shuffled_files = valid_files.copy()
        random.shuffle(shuffled_files)
        
        # Step 2: Initialize request buckets
        requests = [
            {
                "files": [],
                "total_bytes": 0,
                "total_duration": 0.0,
                "file_count": 0
            }
            for _ in range(self.num_requests)
        ]
        
        # Step 3: Round-robin deal into buckets
        for i, file_info in enumerate(shuffled_files):
            bucket_idx = i % self.num_requests
            
            requests[bucket_idx]["files"].append(file_info)
            requests[bucket_idx]["total_bytes"] += file_info["bytes"]
            requests[bucket_idx]["total_duration"] += file_info["duration_sec"]
            requests[bucket_idx]["file_count"] += 1
            
        # Step 4: Optional greedy rebalancing
        self._rebalance_requests(requests)
        
        # Log final balance
        self._log_partition_stats(requests)
        
        return requests
        
    def _rebalance_requests(self, requests: List[Dict], max_iterations: int = 10):
        """
        Perform greedy rebalancing to improve load distribution.
        
        Args:
            requests: List of request buckets to rebalance
            max_iterations: Maximum number of rebalancing iterations
        """
        logger.info("Performing greedy rebalancing...")
        
        for iteration in range(max_iterations):
            improved = False
            
            # Calculate current imbalances
            durations = [req["total_duration"] for req in requests]
            bytes_totals = [req["total_bytes"] for req in requests]
            
            min_duration_idx = durations.index(min(durations))
            max_duration_idx = durations.index(max(durations))
            
            # Skip if already well balanced (within 10%)
            duration_imbalance = (max(durations) - min(durations)) / max(durations)
            if duration_imbalance < 0.1:
                break
                
            # Find a file in the heaviest bucket that we can move to lightest
            max_bucket = requests[max_duration_idx]
            min_bucket = requests[min_duration_idx]
            
            # Sort files in max bucket by duration (prefer moving longer files)
            files_by_duration = sorted(
                max_bucket["files"], 
                key=lambda f: f["duration_sec"], 
                reverse=True
            )
            
            # Try to move the best file
            for file_info in files_by_duration:
                # Check if moving this file improves balance
                new_max_duration = max_bucket["total_duration"] - file_info["duration_sec"]
                new_min_duration = min_bucket["total_duration"] + file_info["duration_sec"]
                
                # Only move if it improves the max-min gap
                if abs(new_max_duration - new_min_duration) < abs(max_bucket["total_duration"] - min_bucket["total_duration"]):
                    # Move the file
                    max_bucket["files"].remove(file_info)
                    max_bucket["total_duration"] -= file_info["duration_sec"]
                    max_bucket["total_bytes"] -= file_info["bytes"]
                    max_bucket["file_count"] -= 1
                    
                    min_bucket["files"].append(file_info)
                    min_bucket["total_duration"] += file_info["duration_sec"]
                    min_bucket["total_bytes"] += file_info["bytes"]
                    min_bucket["file_count"] += 1
                    
                    improved = True
                    break
                    
            if not improved:
                break
                
        logger.info(f"Rebalancing completed after {iteration + 1} iterations")
        
    def _log_partition_stats(self, requests: List[Dict]):
        """Log statistics about the partitioning quality."""
        
        durations = [req["total_duration"] for req in requests]
        bytes_totals = [req["total_bytes"] for req in requests]
        file_counts = [req["file_count"] for req in requests]
        
        logger.info("Partition statistics:")
        logger.info(f"  Duration: avg={sum(durations)/len(durations):.1f}s, "
                   f"min={min(durations):.1f}s, max={max(durations):.1f}s, "
                   f"imbalance={(max(durations)-min(durations))/max(durations)*100:.1f}%")
        
        logger.info(f"  Bytes: avg={sum(bytes_totals)/len(bytes_totals)/1024**2:.1f}MB, "
                   f"min={min(bytes_totals)/1024**2:.1f}MB, max={max(bytes_totals)/1024**2:.1f}MB")
        
        logger.info(f"  Files: avg={sum(file_counts)/len(file_counts):.1f}, "
                   f"min={min(file_counts)}, max={max(file_counts)}")
                   
        # Detailed per-request breakdown
        for i, req in enumerate(requests):
            logger.info(f"  Request {i}: {req['file_count']} files, "
                       f"{req['total_duration']:.1f}s, "
                       f"{req['total_bytes']/1024**2:.1f}MB")


def load_index(index_path: str) -> List[Dict]:
    """
    Load index data from a JSONL file.
    
    Args:
        index_path: Path to the index file
        
    Returns:
        List of file metadata dictionaries
    """
    index_data = []
    
    with open(index_path, 'r') as f:
        for line in f:
            index_data.append(json.loads(line.strip()))
            
    return index_data


def load_requests(requests_path: str) -> List[Dict]:
    """
    Load request partition data from a JSONL file.
    
    Args:
        requests_path: Path to the requests file
        
    Returns:
        List of request dictionaries with file assignments
    """
    requests = []
    
    with open(requests_path, 'r') as f:
        for line in f:
            requests.append(json.loads(line.strip()))
            
    return requests
