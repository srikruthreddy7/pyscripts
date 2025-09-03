"""
ASR (Automatic Speech Recognition) service using nvidia/parakeet-tdt-0.6b-v2.

This service handles:
1. Loading and initializing the Parakeet model
2. Batch processing of audio files for maximum throughput
3. Extracting word-level and segment-level timestamps
4. Parallel file downloads and local caching

Note: Heavy ML deps (torch/torchaudio/transformers/NeMo) are imported lazily
inside methods so that local CLI commands that don't need them can run without
having these packages installed locally. Modal will install them in the image.
"""

import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional
import logging

from typing import Any

from .config import (
    MAX_CONCURRENT_DOWNLOADS,
    TEMP_DIR_PREFIX,
    CACHE_ENV_VARS,
    NEMO_BATCH_SIZE_MULTIPLIER,
    TORCH_COMPILE_ENABLE,
    MIXED_PRECISION,
)

logger = logging.getLogger(__name__)

class ASRService:
    """Service for batch ASR processing using Parakeet."""
    
    def __init__(self, model_id: str, batch_size: int, cache_dir: str):
        """
        Initialize the ASR service.
        
        Args:
            model_id: Hugging Face model ID (nvidia/parakeet-tdt-0.6b-v2)
            batch_size: Batch size for GPU processing
            cache_dir: Directory for model caching
        """
        self.model_id = model_id
        self.batch_size = int(batch_size * NEMO_BATCH_SIZE_MULTIPLIER)
        self.cache_dir = cache_dir
        self.model = None
        # self.ctc_model: Optional external CTC model (not used in baseline path)
        # Resolve device lazily to avoid importing torch at module import time
        try:
            import torch  # type: ignore
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"
        
        # Set up caching environment
        for env_var, path in CACHE_ENV_VARS.items():
            os.environ[env_var] = path
            
        # Initialize model
        self._load_model()
        
    def _load_model(self):
        """Load and initialize the Parakeet model."""
        logger.info(f"Loading ASR model: {self.model_id}")
        
        try:
            # Load the Parakeet model using NeMo
            # Parakeet models are typically available through NeMo
            import nemo.collections.asr as nemo_asr  # type: ignore
            import torch  # type: ignore
            self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                model_name=self.model_id,
                map_location=self.device
            )
            
            # Enable mixed precision if available
            if MIXED_PRECISION and self.device == "cuda":
                self.model = self.model.half()
                
            # Enable torch.compile for performance (if supported)
            if TORCH_COMPILE_ENABLE and hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model)
                    logger.info("Enabled torch.compile for ASR model")
                except Exception as e:
                    logger.warning(f"Could not enable torch.compile: {e}")
            
            self.model.eval()
            # Leave RNNT decoding config as model default; we'll request timestamps per-file during transcribe.

            logger.info(f"ASR model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            # Fallback to Transformers implementation
            logger.info("Falling back to Transformers implementation...")
            self._load_transformers_model()
            
    def _load_transformers_model(self):
        """Load model using Transformers library as fallback."""
        try:
            import torch  # type: ignore
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor  # type: ignore

            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=f"{self.cache_dir}/transformers"
            )
            
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                cache_dir=f"{self.cache_dir}/transformers",
                torch_dtype=torch.float16 if MIXED_PRECISION else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            self.model.eval()
            self.use_transformers = True
            logger.info("Loaded ASR model via Transformers")
            
        except Exception as e:
            logger.error(f"Failed to load model via Transformers: {e}")
            raise
            
    def _download_files_parallel(self, files: List[Dict], temp_dir: str) -> List[str]:
        """
        Download files to local storage in parallel.
        
        Args:
            files: List of file metadata dictionaries
            temp_dir: Temporary directory for downloads
            
        Returns:
            List of local file paths
        """
        logger.info(f"Downloading {len(files)} files to {temp_dir}")
        
        def download_file(file_info: Dict) -> str:
            """Download a single file."""
            src_path = file_info["abspath"]
            dst_path = os.path.join(temp_dir, file_info["relpath"])
            
            # Create directory structure
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # Copy file
            shutil.copy2(src_path, dst_path)
            return dst_path
            
        # Download files in parallel
        local_paths = []
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
            future_to_file = {
                executor.submit(download_file, file_info): file_info 
                for file_info in files
            }
            
            for future in future_to_file:
                try:
                    local_path = future.result(timeout=300)  # 5 minute timeout per file
                    local_paths.append(local_path)
                except Exception as e:
                    file_info = future_to_file[future]
                    logger.error(f"Failed to download {file_info['relpath']}: {e}")
                    
        logger.info(f"Downloaded {len(local_paths)}/{len(files)} files successfully")
        return local_paths
        
    def _load_audio(self, file_path: str) -> Any:
        """
        Load audio file and prepare for ASR.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio tensor ready for processing
        """
        try:
            # Load audio using torchaudio
            import torchaudio  # type: ignore
            import torch  # type: ignore
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Resample if necessary (Parakeet expects 16kHz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                
            return waveform.squeeze(0)  # Remove channel dimension
            
        except Exception as e:
            logger.error(f"Failed to load audio {file_path}: {e}")
            return None
            
    def _transcribe_batch_nemo(self, audio_tensors: List[Any], file_paths: List[str]) -> List[Dict]:
        """
        Transcribe a batch of audio using NeMo model.
        
        Args:
            audio_tensors: List of audio tensors
            file_paths: Corresponding file paths
            
        Returns:
            List of transcription results with timestamps
        """
        try:
            # Convert tensors to numpy arrays for NeMo
            audio_arrays = [tensor.numpy() for tensor in audio_tensors]

            # Transcribe with word-level alignments if supported (baseline: request timestamps for all)
            try:
                results = self.model.transcribe(
                    audio_arrays,
                    batch_size=len(audio_arrays),
                    return_hypotheses=True,
                    timestamps=True,
                )
                logger.debug("Transcription with return_hypotheses succeeded")
            except Exception as e:
                logger.debug(f"return_hypotheses path failed: {e}")
                # Fallback to plain strings
                results = self.model.transcribe(
                    audio_arrays,
                    batch_size=len(audio_arrays)
                )
                logger.debug("Using basic transcription fallback (strings only)")

            # Process results
            transcriptions = []
            for i, (result, file_path) in enumerate(zip(results, file_paths)):
                # Normalize NeMo output: may be a Hypothesis or a list/tuple of Hypotheses
                primary = result[0] if isinstance(result, (list, tuple)) and len(result) > 0 else result

                if isinstance(primary, str):
                    # Simple string result - no timestamps
                    transcription = {
                        "filename": os.path.basename(file_path),
                        "text": primary,
                        "segments": [{"start": 0, "end": duration_sec, "text": primary}],
                        "words": []
                    }
                elif hasattr(primary, 'text'):
                    # NeMo Hypothesis object - extract text and word timestamps
                    words = []
                    segments = []
                    
                    # Try multiple methods to extract word-level information from Hypothesis object
                    logger.debug(f"Processing Hypothesis object: {type(primary)}")
                    try:
                        logger.debug(f"Available attributes: {dir(primary)}")
                    except Exception:
                        pass
                    
                    # Preferred: Hypothesis.timestamp['word'] with start/end in seconds
                    if hasattr(primary, 'timestamp'):
                        try:
                            ts = getattr(primary, 'timestamp')
                            if isinstance(ts, dict) and ts.get('word'):
                                for w in ts['word']:
                                    wtext = w.get('word', '')
                                    # Use seconds if present, else fallback to offsets (approximate)
                                    if 'start' in w and 'end' in w:
                                        ws = float(w.get('start') or 0)
                                        we = float(w.get('end') or ws)
                                    else:
                                        # If only offsets are present, keep as 0 for now
                                        ws = float(w.get('start_offset', 0) or 0)
                                        we = float(w.get('end_offset', ws) or ws)
                                    words.append({"w": wtext, "s": ws, "e": we})
                        except Exception as e:
                            logger.debug(f"No usable Hypothesis.timestamp found: {e}")

                    # Method 1: NeMo word_timestamps (most common)
                    if hasattr(primary, 'word_timestamps') and getattr(primary, 'word_timestamps'):
                        wt = getattr(primary, 'word_timestamps')
                        logger.debug(f"Found word_timestamps with {len(wt)} entries")
                        for w in wt:
                            try:
                                # Support dict-like entries
                                if isinstance(w, dict):
                                    wtext = w.get('word', '')
                                    ws = float(w.get('start_time', w.get('start_offset', 0)) or 0)
                                    we = float(w.get('end_time', w.get('end_offset', ws)) or ws)
                                else:
                                    # Attribute-style entries
                                    wtext = getattr(w, 'word', '')
                                    ws = float(getattr(w, 'start_time', getattr(w, 'start_offset', 0)) or 0)
                                    we = float(getattr(w, 'end_time', getattr(w, 'end_offset', ws)) or ws)
                                words.append({"w": wtext, "s": ws, "e": we})
                            except Exception as e:
                                logger.debug(f"Error extracting word_timestamps entry: {e}")

                    # Method 1b: Alternate naming 'word_ts'
                    if not words and hasattr(primary, 'word_ts') and getattr(primary, 'word_ts'):
                        wt = getattr(primary, 'word_ts')
                        logger.debug(f"Found word_ts with {len(wt)} entries")
                        for w in wt:
                            try:
                                if isinstance(w, dict):
                                    wtext = w.get('word', '')
                                    ws = float(w.get('start_time', w.get('start', 0)) or 0)
                                    we = float(w.get('end_time', w.get('end', ws)) or ws)
                                else:
                                    wtext = getattr(w, 'word', '')
                                    ws = float(getattr(w, 'start_time', getattr(w, 'start', 0)) or 0)
                                    we = float(getattr(w, 'end_time', getattr(w, 'end', ws)) or ws)
                                words.append({"w": wtext, "s": ws, "e": we})
                            except Exception as e:
                                logger.debug(f"Error extracting word_ts entry: {e}")

                    # Method 2: Direct words attribute
                    if hasattr(primary, 'words') and getattr(primary, 'words'):
                        pw = getattr(primary, 'words')
                        logger.debug(f"Found words attribute with {len(pw)} words")
                        for word_info in pw:
                            try:
                                word_data = {}
                                # Try different attribute names for word text
                                if hasattr(word_info, 'word'):
                                    word_data['w'] = word_info.word
                                elif hasattr(word_info, 'text'):
                                    word_data['w'] = word_info.text
                                elif isinstance(word_info, str):
                                    word_data['w'] = word_info
                                    
                                # Try different attribute names for timing
                                if hasattr(word_info, 'start_time'):
                                    word_data['s'] = float(word_info.start_time)
                                elif hasattr(word_info, 'start'):
                                    word_data['s'] = float(word_info.start)
                                    
                                if hasattr(word_info, 'end_time'):
                                    word_data['e'] = float(word_info.end_time)
                                elif hasattr(word_info, 'end'):
                                    word_data['e'] = float(word_info.end)
                                
                                if 'w' in word_data and 's' in word_data and 'e' in word_data:
                                    words.append(word_data)
                                    logger.debug(f"Extracted word: {word_data}")
                            except Exception as e:
                                logger.debug(f"Error extracting word info: {e}")
                    
                    # Method 3: timestep attribute (NeMo specific)
                    elif not words and hasattr(primary, 'timestep') and getattr(primary, 'timestep'):
                        ts = getattr(primary, 'timestep')
                        logger.debug(f"Found timestep attribute: {ts}")
                        if isinstance(ts, dict) and 'word' in ts:
                            word_timestamps = ts['word']
                            for word_info in word_timestamps:
                                try:
                                    words.append({
                                        "w": word_info.get('word', word_info.get('text', '')),
                                        "s": float(word_info.get('start', word_info.get('start_time', 0))),
                                        "e": float(word_info.get('end', word_info.get('end_time', 0)))
                                    })
                                except Exception as e:
                                    logger.debug(f"Error extracting from timestep: {e}")

                    # Method 4: alignments attribute
                    elif not words and hasattr(primary, 'alignments') and getattr(primary, 'alignments'):
                        al = getattr(primary, 'alignments')
                        logger.debug(f"Found alignments attribute with {len(al)} items")
                        # This would need character-to-word grouping logic
                        # For now, just log what's available
                        try:
                            alignment_sample = al[0] if al else None
                            if alignment_sample:
                                logger.debug(f"Sample alignment attributes: {dir(alignment_sample)}")
                        except Exception as e:
                            logger.debug(f"Could not inspect alignments: {e}")
                    
                    if words:
                        logger.info(f"Successfully extracted {len(words)} word timestamps")

                    # If no word-level info, create basic segment
                    if not words:
                        duration = len(audio_tensors[i]) / 16000
                        segments = [{"start": 0, "end": duration, "text": getattr(primary, 'text', '')}]
                    else:
                        # Create segments from words (group consecutive words)
                        if words:
                            segments = [{"start": words[0]["s"], "end": words[-1]["e"], "text": getattr(primary, 'text', '')}]
                    
                    transcription = {
                        "filename": os.path.basename(file_path),
                        "text": getattr(primary, 'text', ''),
                        "segments": segments if segments else [{"start": 0, "end": len(audio_tensors[i]) / 16000, "text": getattr(primary, 'text', '')}],
                        "words": words
                    }
                else:
                    # Rich result with timestamps (dict format)
                    transcription = self._parse_nemo_result(result, file_path)
                    
                transcriptions.append(transcription)
                
            return transcriptions
            
        except Exception as e:
            logger.error(f"Batch transcription failed: {e}")
            return []
            
    def _transcribe_batch_transformers(self, audio_tensors: List[Any], file_paths: List[str]) -> List[Dict]:
        """
        Transcribe a batch of audio using Transformers model.
        
        Args:
            audio_tensors: List of audio tensors
            file_paths: Corresponding file paths
            
        Returns:
            List of transcription results with timestamps
        """
        try:
            # Prepare inputs
            import torch  # type: ignore
            inputs = self.processor(
                [tensor.numpy() for tensor in audio_tensors],
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate transcriptions
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=500
                )
                
            # Decode results
            transcriptions = []
            for i, (generated_id, file_path) in enumerate(zip(generated_ids, file_paths)):
                result = self.processor.decode(
                    generated_id, 
                    skip_special_tokens=True, 
                    decode_with_timestamps=True
                )
                
                transcription = self._parse_transformers_result(result, file_path, audio_tensors[i])
                transcriptions.append(transcription)
                
            return transcriptions
            
        except Exception as e:
            logger.error(f"Batch transcription failed: {e}")
            return []
            
    def _parse_nemo_result(self, result: Dict, file_path: str) -> Dict:
        """Parse NeMo transcription result into standard format."""
        segments = []
        words = []
        
        if "segments" in result:
            for seg in result["segments"]:
                segments.append({
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "")
                })
                
        if "words" in result:
            for word in result["words"]:
                words.append({
                    "w": word.get("word", ""),
                    "s": word.get("start", 0),
                    "e": word.get("end", 0)
                })
                
        return {
            "filename": os.path.basename(file_path),
            "text": result.get("text", ""),
            "segments": segments,
            "words": words
        }
        
    def _parse_transformers_result(self, result: str, file_path: str, audio_tensor: Any) -> Dict:
        """Parse Transformers transcription result into standard format."""
        # For Transformers results, we need to parse timestamp format
        # This is a simplified parser - in practice, you'd want more robust parsing
        
        duration = len(audio_tensor) / 16000
        
        return {
            "filename": os.path.basename(file_path),
            "text": result,
            "segments": [{"start": 0, "end": duration, "text": result}],
            "words": []  # Word-level timestamps would need more complex parsing
        }
        
    def process_files(self, files: List[Dict], output_dir: str, request_id: Optional[int] = None) -> List[Dict]:
        """
        Process a list of files with batch ASR.
        
        Args:
            files: List of file metadata dictionaries
            output_dir: Directory to save interim results
            
        Returns:
            List of transcription results
        """
        logger.info(f"Processing {len(files)} files for ASR")
        
        # Create temporary directory for file downloads
        temp_dir = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
        all_results = []
        
        try:
            # Download files to local storage
            local_paths = self._download_files_parallel(files, temp_dir)
            
            # Sort files by duration for optimal batching
            file_duration_pairs = []
            for local_path in local_paths:
                # Find corresponding file info
                file_info = next(f for f in files if local_path.endswith(f["relpath"]))
                file_duration_pairs.append((local_path, file_info["duration_sec"], file_info))
                
            # Sort by duration (ascending) for better batching
            file_duration_pairs.sort(key=lambda x: x[1])
            
            # Process in batches
            for i in range(0, len(file_duration_pairs), self.batch_size):
                batch_end = min(i + self.batch_size, len(file_duration_pairs))
                batch_files = file_duration_pairs[i:batch_end]
                
                logger.info(f"Processing batch {i//self.batch_size + 1}: files {i+1}-{batch_end}")
                
                # Load audio for this batch
                audio_tensors = []
                batch_paths = []
                
                for file_path, duration, file_info in batch_files:
                    audio_tensor = self._load_audio(file_path)
                    if audio_tensor is not None:
                        audio_tensors.append(audio_tensor)
                        batch_paths.append(file_path)
                        
                if not audio_tensors:
                    logger.warning(f"No valid audio in batch {i//self.batch_size + 1}")
                    continue
                    
                # Transcribe batch
                start_time = time.time()
                
                if hasattr(self, 'use_transformers') and self.use_transformers:
                    batch_results = self._transcribe_batch_transformers(audio_tensors, batch_paths)
                else:
                    batch_results = self._transcribe_batch_nemo(audio_tensors, batch_paths)
                    
                batch_time = time.time() - start_time
                
                logger.info(f"Batch {i//self.batch_size + 1} completed in {batch_time:.1f}s "
                           f"({len(audio_tensors)} files)")
                
                all_results.extend(batch_results)
                
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        # Save interim results
        os.makedirs(output_dir, exist_ok=True)
        
        # Save per-request results (use explicit request_id when provided)
        rid = 0 if request_id is None else request_id
        output_path = os.path.join(output_dir, f"request_{rid}.jsonl")
        
        with open(output_path, 'w') as f:
            for result in all_results:
                f.write(json.dumps(result) + '\n')
                
        logger.info(f"Saved ASR results for {len(all_results)} files to {output_path}")
        
        return all_results
