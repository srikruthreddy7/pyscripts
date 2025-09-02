"""
Utility functions for the batch diarization pipeline.

This module provides:
1. Logging setup and configuration
2. Cost estimation and reporting
3. Performance monitoring and metrics
4. File and path utilities
5. Time and format conversions
"""

import csv
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

from .config import (
    LOG_LEVEL,
    LOG_FORMAT,
    L40S_HOURLY_RATE,
    COST_OVERHEAD_FACTOR,
    DIARIZATION_PRESETS,
    estimate_cost
)

def setup_logging(level: str = LOG_LEVEL, format_str: str = LOG_FORMAT):
    """
    Set up logging configuration for the pipeline.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_str: Log message format string
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("torchaudio").setLevel(logging.WARNING)
    logging.getLogger("nemo").setLevel(logging.WARNING)
    logging.getLogger("pyannote").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at {level} level")

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "2h 34m 56s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.0f}h {minutes:.0f}m"

def format_file_size(bytes_size: int) -> str:
    """
    Format file size in bytes to human-readable string.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.2 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"

def calculate_rtf(processing_time: float, audio_duration: float) -> float:
    """
    Calculate Real Time Factor (RTF).
    
    Args:
        processing_time: Time taken to process in seconds
        audio_duration: Duration of audio in seconds
        
    Returns:
        RTF value (processing_time / audio_duration)
    """
    if audio_duration <= 0:
        return float('inf')
    return processing_time / audio_duration

def generate_cost_report(
    job_id: str, 
    processing_results: List[Dict], 
    preset: str,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a comprehensive cost and performance report.
    
    Args:
        job_id: Job identifier
        processing_results: List of processing results from workers
        preset: Diarization preset used
        output_path: Optional path to save CSV report
        
    Returns:
        Path to generated report file
    """
    logger = logging.getLogger(__name__)
    
    # Calculate totals
    total_files = sum(r.get("files_processed", 0) for r in processing_results)
    total_gpu_hours = sum(r.get("gpu_hours", 0) for r in processing_results)
    total_processing_time = sum(r.get("processing_time_sec", 0) for r in processing_results)
    
    # Estimate costs
    gpu_cost = total_gpu_hours * L40S_HOURLY_RATE
    total_cost = gpu_cost * COST_OVERHEAD_FACTOR
    
    # Get preset info
    preset_config = DIARIZATION_PRESETS.get(preset, {})
    target_rtf = preset_config.get("rtf", 0)
    
    # Generate report
    report_data = {
        "job_info": {
            "job_id": job_id,
            "timestamp": datetime.now().isoformat(),
            "preset": preset,
            "target_rtf": target_rtf
        },
        "processing_stats": {
            "total_files": total_files,
            "total_requests": len(processing_results),
            "total_gpu_hours": round(total_gpu_hours, 3),
            "total_processing_time_sec": round(total_processing_time, 1),
            "avg_processing_time_per_request": round(total_processing_time / len(processing_results), 1) if processing_results else 0
        },
        "cost_breakdown": {
            "gpu_cost_usd": round(gpu_cost, 2),
            "overhead_factor": COST_OVERHEAD_FACTOR,
            "total_cost_usd": round(total_cost, 2),
            "l40s_hourly_rate": L40S_HOURLY_RATE
        },
        "performance_metrics": {
            "files_per_gpu_hour": round(total_files / total_gpu_hours, 1) if total_gpu_hours > 0 else 0,
            "cost_per_file_usd": round(total_cost / total_files, 4) if total_files > 0 else 0
        }
    }
    
    # Save detailed CSV report if path provided
    if output_path is None:
        output_path = f"/vol/results/cost_report_{job_id}.csv"
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow([
            "request_id", "files_processed", "processing_time_sec", 
            "gpu_hours", "estimated_cost_usd", "preset"
        ])
        
        # Write per-request data
        for i, result in enumerate(processing_results):
            request_cost = result.get("gpu_hours", 0) * L40S_HOURLY_RATE * COST_OVERHEAD_FACTOR
            writer.writerow([
                result.get("request_id", i),
                result.get("files_processed", 0),
                round(result.get("processing_time_sec", 0), 1),
                round(result.get("gpu_hours", 0), 3),
                round(request_cost, 2),
                result.get("preset_used", preset)
            ])
            
        # Write summary row
        writer.writerow([])
        writer.writerow(["TOTAL", total_files, round(total_processing_time, 1), 
                        round(total_gpu_hours, 3), round(total_cost, 2), preset])
        
    # Save JSON summary
    json_path = output_path.replace('.csv', '.json')
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2)
        
    logger.info(f"Cost report saved to: {output_path}")
    logger.info(f"Summary JSON saved to: {json_path}")
    
    # Log summary
    logger.info("=" * 60)
    logger.info(f"COST REPORT - JOB: {job_id}")
    logger.info("=" * 60)
    logger.info(f"Files processed: {total_files:,}")
    logger.info(f"Total GPU hours: {total_gpu_hours:.3f}")
    logger.info(f"Total cost: ${total_cost:.2f}")
    logger.info(f"Cost per file: ${total_cost/total_files:.4f}" if total_files > 0 else "Cost per file: N/A")
    logger.info(f"Files per GPU hour: {total_files/total_gpu_hours:.1f}" if total_gpu_hours > 0 else "Files per GPU hour: N/A")
    logger.info("=" * 60)
    
    return output_path

class PerformanceMonitor:
    """Monitor and track performance metrics during processing."""
    
    def __init__(self, job_id: str):
        """
        Initialize performance monitor.
        
        Args:
            job_id: Job identifier for tracking
        """
        self.job_id = job_id
        self.start_time = time.time()
        self.metrics = {
            "files_processed": 0,
            "total_audio_duration": 0,
            "total_processing_time": 0,
            "gpu_time": 0,
            "rtf_values": [],
            "errors": []
        }
        
    def record_file_processing(
        self, 
        file_path: str, 
        audio_duration: float, 
        processing_time: float,
        success: bool = True,
        error_msg: Optional[str] = None
    ):
        """
        Record metrics for a processed file.
        
        Args:
            file_path: Path to processed file
            audio_duration: Duration of audio in seconds
            processing_time: Time taken to process in seconds
            success: Whether processing was successful
            error_msg: Error message if processing failed
        """
        self.metrics["files_processed"] += 1
        
        if success:
            self.metrics["total_audio_duration"] += audio_duration
            self.metrics["total_processing_time"] += processing_time
            
            rtf = calculate_rtf(processing_time, audio_duration)
            self.metrics["rtf_values"].append(rtf)
        else:
            self.metrics["errors"].append({
                "file": file_path,
                "error": error_msg,
                "timestamp": time.time()
            })
            
    def get_current_stats(self) -> Dict:
        """
        Get current performance statistics.
        
        Returns:
            Dict with current performance metrics
        """
        elapsed_time = time.time() - self.start_time
        rtf_values = self.metrics["rtf_values"]
        
        stats = {
            "elapsed_time_sec": elapsed_time,
            "files_processed": self.metrics["files_processed"],
            "total_audio_hours": self.metrics["total_audio_duration"] / 3600,
            "total_processing_time_sec": self.metrics["total_processing_time"],
            "error_count": len(self.metrics["errors"]),
            "success_rate": (self.metrics["files_processed"] - len(self.metrics["errors"])) / max(1, self.metrics["files_processed"]),
            "avg_rtf": sum(rtf_values) / len(rtf_values) if rtf_values else 0,
            "median_rtf": sorted(rtf_values)[len(rtf_values)//2] if rtf_values else 0,
            "files_per_hour": self.metrics["files_processed"] / (elapsed_time / 3600) if elapsed_time > 0 else 0
        }
        
        return stats
        
    def print_progress_report(self):
        """Print a progress report to the log."""
        logger = logging.getLogger(__name__)
        stats = self.get_current_stats()
        
        logger.info("=" * 50)
        logger.info(f"PROGRESS REPORT - JOB: {self.job_id}")
        logger.info("=" * 50)
        logger.info(f"Elapsed time: {format_duration(stats['elapsed_time_sec'])}")
        logger.info(f"Files processed: {stats['files_processed']:,}")
        logger.info(f"Audio processed: {stats['total_audio_hours']:.1f} hours")
        logger.info(f"Success rate: {stats['success_rate']*100:.1f}%")
        logger.info(f"Average RTF: {stats['avg_rtf']:.4f}")
        logger.info(f"Processing rate: {stats['files_per_hour']:.1f} files/hour")
        
        if stats['error_count'] > 0:
            logger.warning(f"Errors encountered: {stats['error_count']}")
            
        logger.info("=" * 50)

def validate_audio_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate an audio file for processing.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return False, "File does not exist"
            
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "File is empty"
        elif file_size > 2 * 1024**3:  # 2GB limit
            return False, f"File too large: {format_file_size(file_size)}"
            
        # Check file extension
        valid_extensions = ['.flac', '.wav', '.mp3', '.m4a']
        if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
            return False, f"Unsupported file format"
            
        return True, None
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def create_output_directories(job_id: str, base_dir: str = "/vol/results") -> Dict[str, str]:
    """
    Create all necessary output directories for a job.
    
    Args:
        job_id: Job identifier
        base_dir: Base directory for results
        
    Returns:
        Dict mapping directory types to paths
    """
    directories = {
        "base": base_dir,
        "jsonl": os.path.join(base_dir, "jsonl", job_id),
        "srt": os.path.join(base_dir, "srt", job_id),
        "interim": os.path.join(base_dir, "interim", job_id),
        "reports": os.path.join(base_dir, "reports", job_id)
    }
    
    # Create all directories
    for dir_type, dir_path in directories.items():
        os.makedirs(dir_path, exist_ok=True)
        
    return directories

def cleanup_temp_files(temp_dirs: List[str]):
    """
    Clean up temporary directories and files.
    
    Args:
        temp_dirs: List of temporary directory paths to clean up
    """
    logger = logging.getLogger(__name__)
    
    for temp_dir in temp_dirs:
        try:
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up {temp_dir}: {e}")

def estimate_job_duration(
    total_audio_hours: float, 
    num_requests: int, 
    preset: str = "high_latency"
) -> Dict[str, float]:
    """
    Estimate how long a job will take to complete.
    
    Args:
        total_audio_hours: Total hours of audio to process
        num_requests: Number of parallel requests
        preset: Diarization preset
        
    Returns:
        Dict with time estimates
    """
    preset_config = DIARIZATION_PRESETS.get(preset, {})
    rtf = preset_config.get("rtf", 0.005)
    
    # Sequential processing time
    sequential_time_hours = total_audio_hours * rtf
    
    # Parallel processing time (assuming perfect load balancing)
    parallel_time_hours = sequential_time_hours / num_requests
    
    # Add overhead for startup, file transfer, etc. (20%)
    estimated_wall_clock_hours = parallel_time_hours * 1.2
    
    return {
        "sequential_hours": sequential_time_hours,
        "parallel_hours": parallel_time_hours,
        "estimated_wall_clock_hours": estimated_wall_clock_hours,
        "estimated_wall_clock_minutes": estimated_wall_clock_hours * 60,
        "rtf": rtf,
        "parallelism": num_requests
    }


