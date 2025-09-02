"""
Batch Audio Transcription and Diarization Pipeline

A high-throughput Modal-based pipeline for processing large-scale audio datasets
with speaker diarization using open-source models.

Author: Assistant
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Assistant"
__license__ = "MIT"

# Import main components for easy access
from .config import (
    ASR_MODEL_ID,
    DIARIZATION_MODEL_ID,
    DIARIZATION_PRESETS,
    estimate_cost,
    get_recommended_preset
)

from .utils import (
    setup_logging,
    format_duration,
    format_file_size,
    calculate_rtf,
    generate_cost_report
)

__all__ = [
    "ASR_MODEL_ID",
    "DIARIZATION_MODEL_ID", 
    "DIARIZATION_PRESETS",
    "estimate_cost",
    "get_recommended_preset",
    "setup_logging",
    "format_duration",
    "format_file_size",
    "calculate_rtf",
    "generate_cost_report"
]



