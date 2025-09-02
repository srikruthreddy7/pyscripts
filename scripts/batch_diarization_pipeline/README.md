# Batch Audio Transcription and Diarization Pipeline

A high-throughput Modal-based pipeline for processing large-scale audio datasets with speaker diarization. Processes ~467 hours of 16kHz FLAC audio for under $10 using open-source models.

## Features

- **Scalable Processing**: Handles hundreds of hours of audio with parallel GPU workers
- **Open Source Models**: Uses nvidia/parakeet-tdt-0.6b-v2 (ASR) and nvidia/diar_streaming_sortformer_4spk-v2 (diarization)
- **Cost Optimized**: Smart batching and throughput presets keep costs under $10 for ~467 hours
- **Speaker Labels**: Produces speaker-labeled transcripts with word-level timestamps
- **Multiple Outputs**: Generates JSON and SRT files with comprehensive metadata

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Audio Files   │───▶│   Indexing &     │───▶│   Partitioning  │
│   (~790 FLAC)   │    │   Validation     │    │   (25 requests) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Final Results  │◀───│     Fusion       │◀───│  ASR + Diarize  │
│ (JSON + SRT)    │    │   (Alignment)    │    │   (Parallel)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Setup

```bash
# Install Modal CLI
pip install modal

# Clone/download this pipeline
cd batch_diarization_pipeline

# Install dependencies (handled automatically by Modal)
# Requirements are defined in config.py's get_modal_image()
```

### 2. Prepare Audio Data

Input Data in Modal Volume

If your data already lives in a Modal Volume named `lynkup-audio-volume` under a subfolder like `2025/08/28`, you do not need to re-upload. The pipeline mounts that volume at `/vol/audio` inside the container, so your files will appear at `/vol/audio/2025/08/28`.

Only upload if you need to add files:

```bash
# Create and mount the volume (one-time setup)
modal volume create lynkup-audio-volume
modal volume put lynkup-audio-volume /path/to/your/audio/files /vol/audio/
```

### 3. Run the Pipeline

Optionally verify inputs (counts and size):
```bash
modal run app.py::verify_inputs --audio-subdir 2025/08/28
```

Estimate runtime and cost (exact):
```bash
modal run app.py::estimate_runtime_and_cost \
  --job-id hvac-2025-01 \
  --preset high_latency \
  --num-requests 25 \
  --audio-subdir 2025/08/28
```

**Option A: Full Pipeline (Recommended)**
```bash
modal run app.py::full_pipeline --job-id hvac-2025-01 --preset high_latency --num-requests 25 --audio-subdir 2025/08/28
```

**Option B: Step-by-Step Execution**
```bash
# Step 1: ASR Transcription
modal run app.py::batch_transcription --job-id hvac-2025-01 --num-requests 25 --audio-subdir 2025/08/28

# Step 2: Diarization + Fusion  
modal run app.py::batch_diarize_and_fuse --job-id hvac-2025-01 --preset high_latency
```

### 4. Retrieve Results

```bash
# Download results from Modal Volume
modal volume get results /vol/results/final_hvac-2025-01/ ./results/
modal volume get results /vol/results/srt_hvac-2025-01/ ./srt_files/
```

## Configuration

### Diarization Presets

Choose a preset based on your budget and quality requirements:

| Preset | RTF | Cost (467h) | Quality | Use Case |
|--------|-----|-------------|---------|-----------|
| `very_high_latency` | 0.002 | ~$1.82 | Highest | Large batches, tight budget |
| `high_latency` | 0.005 | ~$4.55 | High | **Recommended for most use cases** |
| `low_latency` | 0.093 | ~$84.50 | Medium | Not recommended for batch |
| `ultra_low_latency` | 0.180 | ~$163.00 | Medium | Not recommended for batch |

### Key Parameters

```bash
# Core parameters
--job-id JOB_ID                 # Unique identifier for this job
--preset PRESET                 # Diarization preset (high_latency, very_high_latency)
--num-requests NUM              # Parallel workers (default: 25)
--gpu-batch-size SIZE           # ASR batch size (default: 128)

# Advanced parameters  
--model-id MODEL                # ASR model (default: nvidia/parakeet-tdt-0.6b-v2)
--gpu-type GPU                  # GPU type (default: L40S)
```

## Output Format

### JSON Output Schema

Each audio file produces a JSON file with this structure:

```json
{
  "filename": "call_001.flac",
  "duration_sec": 1847.3,
  "sample_rate": 16000,
  "speakers": ["SENIOR", "JUNIOR"],
  "segments": [
    {
      "start": 12.34,
      "end": 15.67,
      "speaker": "SENIOR",
      "text": "Okay, check the condenser fan motor first",
      "words": [
        {"w": "Okay", "s": 12.34, "e": 12.55},
        {"w": "check", "s": 12.55, "e": 12.72},
        {"w": "the", "s": 12.72, "e": 12.83},
        {"w": "condenser", "s": 12.83, "e": 13.45},
        {"w": "fan", "s": 13.45, "e": 13.78},
        {"w": "motor", "s": 13.78, "e": 14.12},
        {"w": "first", "s": 14.12, "e": 15.67}
      ]
    }
  ],
  "metadata": {
    "asr_model": "nvidia/parakeet-tdt-0.6b-v2",
    "diarization_model": "nvidia/diar_streaming_sortformer_4spk-v2", 
    "diarization_preset": "high_latency",
    "num_asr_words": 234,
    "num_diar_segments": 12,
    "num_final_segments": 8,
    "rtf": 0.0048
  }
}
```

### SRT Output

Standard subtitle format with speaker labels:

```srt
1
00:00:12,340 --> 00:00:15,670
[SENIOR] Okay, check the condenser fan motor first

2  
00:00:16,120 --> 00:00:19,450
[JUNIOR] I'm looking at it now, seems to be running fine

3
00:00:20,100 --> 00:00:24,230
[SENIOR] Check the capacitor readings then
```

## Volume Structure

The pipeline uses three Modal Volumes:

```
audio-data/          # Input audio files
├── call_001.flac
├── call_002.flac
└── ...

models-cache/        # Cached model weights
├── transformers/
├── nemo/
└── pyannote/

results/             # All outputs and intermediate files
├── index_JOB_ID.jsonl           # File index
├── requests_JOB_ID.jsonl        # Request partitions
├── asr_interim_JOB_ID/          # ASR results
├── final_JOB_ID/                # Final JSON results
├── srt_JOB_ID/                  # SRT subtitle files
└── cost_report_JOB_ID.csv       # Cost and performance report
```

## Cost Estimation

For 467 hours of audio with `high_latency` preset:

```
Diarization: 467h × 0.005 RTF × $1.95/h = $4.55
ASR:         467h × 0.005 RTF × $1.95/h = $2.28  
Overhead:    20% container/transfer costs = $1.37
Total:       ~$8.20 (under $10 target)
```

Use the cost estimator:

```python
from config import estimate_cost

# Estimate cost for your dataset
cost_breakdown = estimate_cost(
    total_audio_hours=467, 
    preset="high_latency"
)
print(f"Estimated cost: ${cost_breakdown['total']['cost_usd_with_overhead']:.2f}")
```

## Performance Tuning

### Optimizing Throughput

1. **Increase Parallelism**: More `--num-requests` workers (limited by Modal quotas)
2. **Tune Batch Size**: Adjust `--gpu-batch-size` based on GPU memory
3. **Choose Faster Preset**: Use `very_high_latency` for maximum throughput

### Monitoring Progress

```bash
# Monitor job progress (live)
modal logs batch-diarization-pipeline

# Check cost report after completion
modal volume get results /vol/results/cost_report_JOB_ID.csv ./
```

### Troubleshooting

**Common Issues:**

1. **Out of Memory**: Reduce `--gpu-batch-size` from 128 to 64 or 32
2. **File Format Errors**: Ensure audio is 16kHz mono FLAC
3. **Missing Models**: First run downloads models (may take 10-15 minutes)
4. **Volume Access**: Ensure volumes are properly mounted with correct permissions

**Debug Mode:**
```bash
# Run with verbose logging
modal run app.py::full_pipeline --job-id debug-test --num-requests 1
```

## Example Workflows

### Small Test Run
```bash
# Test with subset of files and 1 worker
modal run app.py::full_pipeline \
  --job-id test-run-$(date +%Y%m%d) \
  --preset very_high_latency \
  --num-requests 1
```

### Production Run  
```bash
# Full dataset with cost optimization
modal run app.py::full_pipeline \
  --job-id production-$(date +%Y%m%d) \
  --preset high_latency \
  --num-requests 25 \
  --gpu-batch-size 128
```

### Budget-Constrained Run
```bash
# Maximum cost savings
modal run app.py::full_pipeline \
  --job-id budget-run-$(date +%Y%m%d) \
  --preset very_high_latency \
  --num-requests 50
```

## File Organization

```
batch_diarization_pipeline/
├── app.py                    # Main Modal app with entry points
├── config.py                 # Configuration and presets
├── indexer.py               # Audio file indexing and partitioning
├── asr_service.py           # ASR transcription service
├── diarization_service.py   # Speaker diarization service  
├── fusion_service.py        # ASR+diarization fusion
├── utils.py                 # Utilities and cost reporting
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Model Information

### ASR: nvidia/parakeet-tdt-0.6b-v2
- **License**: CC-BY-4.0 (Commercial OK)
- **Language**: English
- **Features**: Word-level timestamps, high accuracy
- **Size**: ~600MB

### Diarization: nvidia/diar_streaming_sortformer_4spk-v2  
- **License**: CC-BY-4.0 (Commercial OK)
- **Speakers**: Up to 4 speakers supported
- **Features**: Configurable latency/throughput tradeoffs
- **Size**: ~150MB

## Advanced Usage

### Custom Speaker Mapping

Modify `fusion_service.py` to customize speaker label assignment:

```python
# In FusionService._determine_speaker_mapping()
# Add custom logic based on your domain knowledge
```

### Processing Non-FLAC Files

Convert audio files before upload:

```bash
# Convert WAV to FLAC
for f in *.wav; do 
    ffmpeg -i "$f" -ar 16000 -ac 1 "${f%.wav}.flac"
done
```

### Batch Validation

Validate files before processing:

```python
from utils import validate_audio_file

for file_path in audio_files:
    is_valid, error = validate_audio_file(file_path)
    if not is_valid:
        print(f"Invalid: {file_path} - {error}")
```

## License

This pipeline is provided under MIT License. Note that the models have their own licenses:
- Parakeet: CC-BY-4.0
- Sortformer v2: CC-BY-4.0

Both models are approved for commercial use.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Modal logs: `modal logs batch-diarization-pipeline`  
3. Validate your audio format and volume setup
4. Consider reducing parallelism for debugging

## References

- [Modal Open Batch Transcription](https://modal.com/docs/examples/open-batch-transcription)
- [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2)
- [Modal GPU Pricing](https://modal.com/pricing)
