"""
Configuration file for the batch diarization pipeline.

Contains model IDs, GPU settings, diarization presets, and Modal image definitions.
"""

import os
import modal

# Model configurations
ASR_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
DIARIZATION_MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2"

# GPU and processing settings
GPU_TYPE = "L40S"
DEFAULT_GPU_BATCH_SIZE = 128
DEFAULT_NUM_REQUESTS = 25

# Audio format requirements
REQUIRED_SAMPLE_RATE = 16000
REQUIRED_CHANNELS = 1
SUPPORTED_FORMATS = [".flac"]

# Diarization presets based on Sortformer v2 documentation
# RTF = Real Time Factor (processing_time / audio_duration)
DIARIZATION_PRESETS = {
    "high_latency": {
        "rtf": 0.005,
        "chunk_size": 124,
        "right_context": 1,
        "fifo_size": 124,
        "update_period": 124,
        "speaker_cache": 188,
        "description": "High latency preset - good balance of cost and quality",
        "recommended_for": "Production batch processing"
    },
    "very_high_latency": {
        "rtf": 0.002,
        "chunk_size": 340,
        "right_context": 40,
        "fifo_size": 40,
        "update_period": 300,
        "speaker_cache": 188,
        "description": "Very high latency preset - lowest cost, highest quality",
        "recommended_for": "Large scale batch processing with cost constraints"
    },
    # These are for streaming/real-time (not recommended for batch)
    "low_latency": {
        "rtf": 0.093,
        "chunk_size": 60,
        "right_context": 1,
        "fifo_size": 60,
        "update_period": 60,
        "speaker_cache": 90,
        "description": "Low latency preset - expensive, for real-time",
        "recommended_for": "Real-time applications only"
    },
    "ultra_low_latency": {
        "rtf": 0.180,
        "chunk_size": 40,
        "right_context": 1,
        "fifo_size": 40,
        "update_period": 40,
        "speaker_cache": 60,
        "description": "Ultra low latency preset - very expensive, for real-time",
        "recommended_for": "Real-time applications with strict latency requirements"
    }
}

# Speaker mapping configuration
SPEAKER_LABELS = ["SUPPORT", "TECH"]
DEFAULT_SPEAKER_MAPPING = {0: "SUPPORT", 1: "JUTECHNIOR"}

# File processing settings
MAX_CONCURRENT_DOWNLOADS = 8
TEMP_DIR_PREFIX = "/tmp/batch_audio_"
SEGMENT_MERGE_GAP_THRESHOLD = 0.3  # seconds

# Output format settings
OUTPUT_FORMATS = ["json", "srt"]
JSONL_INDENT = 2

# Cost estimation (L40S pricing)
L40S_HOURLY_RATE = 1.95  # USD per hour
COST_OVERHEAD_FACTOR = 1.2  # 20% overhead for container startup, etc.

# Modal volume names (overridable via env vars)
AUDIO_VOLUME_NAME = os.getenv("AUDIO_VOLUME_NAME", "lynkup-audio-volume")
MODELS_CACHE_VOLUME_NAME = os.getenv("MODELS_CACHE_VOLUME_NAME", "lynkup-models-cache")
RESULTS_VOLUME_NAME = os.getenv("RESULTS_VOLUME_NAME", "lynkup-results")

def get_modal_image():
    """
    Create and return the Modal image with all required dependencies.
    
    This image includes:
    - PyTorch with CUDA support
    - NeMo toolkit for ASR
    - Transformers and related libraries for diarization
    - Audio processing libraries
    - Other utilities
    """
    img = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            # System dependencies
            "ffmpeg",
            "libsndfile1",
            "libsox-fmt-all",
            "sox",
            "git",
        )
        .pip_install(
            # Core ML libraries
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "torchvision>=0.15.0",
            # NeMo for ASR
            "nemo-toolkit[asr]>=1.22.0",
            # Transformers ecosystem for diarization
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "datasets>=2.12.0",
            # Audio processing
            "librosa>=0.10.0",
            "soundfile>=0.12.0",
            "scipy>=1.10.0",
            # Diarization specific
            "pyannote.audio>=3.1.0",
            "pyannote.core>=5.0.0",
            "speechbrain>=0.5.15",
            # Utilities
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "tqdm>=4.65.0",
            "click>=8.1.0",
            "pydub>=0.25.0",
            "python-dateutil>=2.8.0",
            # Modal specific
            "modal>=0.55.0"
        )
        .pip_install(
            # Install specific versions for compatibility
            "omegaconf>=2.3.0",
            "hydra-core>=1.3.0",
            "lightning>=2.0.0",
        )
        .run_commands(
            # Create cache directories
            "mkdir -p /root/.cache/huggingface",
            "mkdir -p /root/.cache/torch",
            "mkdir -p /tmp/audio_processing",
            # Set environment variables for optimal performance
            "echo 'export TRANSFORMERS_CACHE=/vol/models-cache/transformers' >> /root/.bashrc",
            "echo 'export HF_HOME=/vol/models-cache/huggingface' >> /root/.bashrc",
            "echo 'export TORCH_HOME=/vol/models-cache/torch' >> /root/.bashrc",
        )
    )
    # Include local Python package "scripts" so module-mode imports like
    # `scripts.batch_diarization_pipeline.app` resolve inside the container.
    try:
        img = img.add_local_python_source("scripts")
    except Exception:
        # Fallback for older SDKs: leave as-is; module mode may still mount package automatically
        pass
    return img

def validate_preset(preset: str) -> bool:
    """Validate that the given preset is supported."""
    return preset in DIARIZATION_PRESETS

def get_preset_config(preset: str) -> dict:
    """Get configuration for a specific diarization preset."""
    if not validate_preset(preset):
        raise ValueError(f"Unsupported preset: {preset}. Supported: {list(DIARIZATION_PRESETS.keys())}")
    return DIARIZATION_PRESETS[preset]

def estimate_cost(total_audio_hours: float, preset: str = "high_latency") -> dict:
    """
    Estimate the total cost for processing given hours of audio.
    
    Args:
        total_audio_hours: Total hours of audio to process
        preset: Diarization preset to use
        
    Returns:
        Dict with cost breakdown
    """
    preset_config = get_preset_config(preset)
    rtf = preset_config["rtf"]
    
    # Diarization cost calculation
    diar_gpu_hours = total_audio_hours * rtf
    diar_cost = diar_gpu_hours * L40S_HOURLY_RATE
    
    # ASR cost (much lower, estimated at ~0.5% of audio duration)
    asr_gpu_hours = total_audio_hours * 0.005
    asr_cost = asr_gpu_hours * L40S_HOURLY_RATE
    
    # Total with overhead
    subtotal = diar_cost + asr_cost
    total_cost = subtotal * COST_OVERHEAD_FACTOR
    
    return {
        "total_audio_hours": total_audio_hours,
        "preset": preset,
        "rtf": rtf,
        "diarization": {
            "gpu_hours": diar_gpu_hours,
            "cost_usd": diar_cost
        },
        "asr": {
            "gpu_hours": asr_gpu_hours,
            "cost_usd": asr_cost
        },
        "total": {
            "gpu_hours": diar_gpu_hours + asr_gpu_hours,
            "cost_usd_before_overhead": subtotal,
            "cost_usd_with_overhead": total_cost,
            "overhead_factor": COST_OVERHEAD_FACTOR
        }
    }

def get_recommended_preset(total_audio_hours: float, budget_usd: float = 10.0) -> str:
    """
    Recommend the best preset based on audio duration and budget.
    
    Args:
        total_audio_hours: Total hours of audio to process
        budget_usd: Budget constraint in USD
        
    Returns:
        Recommended preset name
    """
    for preset in ["very_high_latency", "high_latency", "low_latency", "ultra_low_latency"]:
        cost_estimate = estimate_cost(total_audio_hours, preset)
        if cost_estimate["total"]["cost_usd_with_overhead"] <= budget_usd:
            return preset
    
    # If even the cheapest option exceeds budget, still recommend very_high_latency
    return "very_high_latency"

# Environment variables for model caching
CACHE_ENV_VARS = {
    "TRANSFORMERS_CACHE": "/vol/models-cache/transformers",
    "HF_HOME": "/vol/models-cache/huggingface", 
    "TORCH_HOME": "/vol/models-cache/torch",
    "HF_DATASETS_CACHE": "/vol/models-cache/datasets",
    "NEMO_CACHE_DIR": "/vol/models-cache/nemo"
}

# File size and processing limits
MAX_FILE_SIZE_GB = 2.0  # Maximum file size to process
MAX_DURATION_HOURS = 5.0  # Maximum single file duration (increased for 4+ hour files)
CHUNK_DURATION_MINUTES = 30  # For chunking very long files

# Validation thresholds
MIN_AUDIO_DURATION_SEC = 1.0  # Minimum audio duration to process
MAX_AUDIO_DURATION_SEC = MAX_DURATION_HOURS * 3600

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance tuning
NEMO_BATCH_SIZE_MULTIPLIER = 1.0  # Adjust based on GPU memory
TORCH_COMPILE_ENABLE = True  # Enable torch.compile for performance
MIXED_PRECISION = True  # Use mixed precision training/inference
