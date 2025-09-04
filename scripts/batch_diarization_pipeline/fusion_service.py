"""
Fusion service for combining ASR transcription and diarization results.

This service handles:
1. Aligning ASR word timestamps with diarization speaker segments
2. Merging consecutive segments from the same speaker
3. Mapping speaker IDs to human-readable labels (SENIOR/JUNIOR)
4. Generating final JSON and SRT output formats
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import logging

from .config import (
    SPEAKER_LABELS,
    DEFAULT_SPEAKER_MAPPING,
    SEGMENT_MERGE_GAP_THRESHOLD,
    JSONL_INDENT
)

logger = logging.getLogger(__name__)

class FusionService:
    """Service for fusing ASR and diarization results."""
    
    def __init__(self):
        """Initialize the fusion service."""
        self.speaker_mapping = DEFAULT_SPEAKER_MAPPING.copy()
        
    def fuse_results(
        self, 
        asr_result: Dict, 
        diarization_result: Dict, 
        file_info: Dict
    ) -> Dict:
        """
        Fuse ASR transcription with diarization to create speaker-labeled transcript.
        
        Args:
            asr_result: ASR transcription result with words and segments
            diarization_result: Diarization result with speaker segments
            file_info: Original file metadata
            
        Returns:
            Fused result with speaker-labeled segments and words
        """
        logger.info(f"Fusing results for: {file_info['relpath']}")
        
        # Extract data from inputs
        asr_words = asr_result.get("words", [])
        asr_segments = asr_result.get("segments", [])
        diar_segments = diarization_result.get("segments", [])
        
        # If no diarization segments, create a single segment
        if not diar_segments:
            logger.warning(f"No diarization segments found for {file_info['relpath']}")
            diar_segments = [{
                "start": 0,
                "end": file_info["duration_sec"],
                "speaker": 0,
                "confidence": 0.5
            }]
        
        # Determine speaker mapping for this file
        file_speaker_mapping = self._determine_speaker_mapping(diar_segments, asr_words)
        
        # Align words with diarization segments
        aligned_segments = self._align_words_with_speakers(asr_words, diar_segments)
        
        # Merge consecutive segments from same speaker
        merged_segments = self._merge_consecutive_segments(aligned_segments)
        
        # Apply speaker labels
        labeled_segments = self._apply_speaker_labels(merged_segments, file_speaker_mapping)
        
        # Create final result
        result = {
            "filename": file_info["relpath"],
            "duration_sec": file_info["duration_sec"],
            "sample_rate": file_info.get("sample_rate", 16000),
            "speakers": list(file_speaker_mapping.values()),
            "segments": labeled_segments,
            "metadata": {
                "asr_model": asr_result.get("model", "unknown"),
                "diarization_model": diarization_result.get("model", "unknown"),
                "diarization_preset": diarization_result.get("preset", "unknown"),
                "num_asr_words": len(asr_words),
                "num_diar_segments": len(diar_segments),
                "num_final_segments": len(labeled_segments),
                "rtf": diarization_result.get("rtf", 0)
            }
        }
        
        logger.info(f"Fusion complete: {len(labeled_segments)} segments, "
                   f"{len(set(seg['speaker'] for seg in labeled_segments))} speakers")
        
        return result
        
    def _determine_speaker_mapping(self, diar_segments: List[Dict], asr_words: List[Dict]) -> Dict[int, str]:
        """
        Determine speaker ID to label mapping for this file.
        
        Uses heuristics to assign SENIOR/JUNIOR labels:
        1. Speaker who talks first gets JUNIOR (if shorter average utterance)
        2. Otherwise, speaker with longer total talk time gets SENIOR
        
        Args:
            diar_segments: Diarization segments
            asr_words: ASR word list
            
        Returns:
            Mapping from speaker ID to label
        """
        if not diar_segments:
            return {0: "SENIOR"}
            
        # Analyze speaker statistics
        speaker_stats = {}
        
        for segment in diar_segments:
            speaker_id = segment["speaker"]
            duration = segment["end"] - segment["start"]
            
            if speaker_id not in speaker_stats:
                speaker_stats[speaker_id] = {
                    "total_duration": 0,
                    "segment_count": 0,
                    "first_appearance": segment["start"]
                }
                
            speaker_stats[speaker_id]["total_duration"] += duration
            speaker_stats[speaker_id]["segment_count"] += 1
            speaker_stats[speaker_id]["first_appearance"] = min(
                speaker_stats[speaker_id]["first_appearance"],
                segment["start"]
            )
            
        # Calculate average segment duration for each speaker
        for speaker_id in speaker_stats:
            stats = speaker_stats[speaker_id]
            stats["avg_segment_duration"] = stats["total_duration"] / stats["segment_count"]
            
        # Sort speakers by first appearance
        speakers_by_appearance = sorted(
            speaker_stats.keys(),
            key=lambda sid: speaker_stats[sid]["first_appearance"]
        )
        
        # Create mapping
        mapping = {}
        
        if len(speakers_by_appearance) == 1:
            # Only one speaker
            mapping[speakers_by_appearance[0]] = "SENIOR"
        elif len(speakers_by_appearance) >= 2:
            # Two or more speakers
            first_speaker = speakers_by_appearance[0]
            second_speaker = speakers_by_appearance[1]
            
            # Heuristic: if first speaker has shorter average segments, they're JUNIOR
            first_avg = speaker_stats[first_speaker]["avg_segment_duration"]
            second_avg = speaker_stats[second_speaker]["avg_segment_duration"]
            
            if first_avg < second_avg * 0.8:  # 20% threshold
                mapping[first_speaker] = "JUNIOR"
                mapping[second_speaker] = "SENIOR"
            else:
                mapping[first_speaker] = "SENIOR"
                mapping[second_speaker] = "JUNIOR"
                
            # Additional speakers get generic labels
            for i, speaker_id in enumerate(speakers_by_appearance[2:], start=2):
                mapping[speaker_id] = f"SPEAKER_{i+1}"
        
        logger.info(f"Speaker mapping: {mapping}")
        return mapping
        
    def _align_words_with_speakers(
        self, 
        asr_words: List[Dict], 
        diar_segments: List[Dict]
    ) -> List[Dict]:
        """
        Align ASR words with diarization segments to create speaker-labeled segments.
        
        Args:
            asr_words: List of ASR words with timestamps
            diar_segments: List of diarization segments with speaker IDs
            
        Returns:
            List of segments with words aligned to speakers
        """
        if not asr_words or not diar_segments:
            return []
            
        # Sort both lists by start time
        asr_words = sorted(asr_words, key=lambda w: w.get("s", w.get("start", 0)))
        diar_segments = sorted(diar_segments, key=lambda s: s["start"])
        
        aligned_segments = []
        
        for diar_segment in diar_segments:
            diar_start = diar_segment["start"]
            diar_end = diar_segment["end"]
            speaker_id = diar_segment["speaker"]
            
            # Find words that overlap with this diarization segment
            segment_words = []
            segment_text_parts = []
            
            for word in asr_words:
                word_start = word.get("s", word.get("start", 0))
                word_end = word.get("e", word.get("end", word_start))
                word_text = word.get("w", word.get("word", ""))
                
                # Check for overlap
                if self._time_overlap(word_start, word_end, diar_start, diar_end):
                    segment_words.append({
                        "w": word_text,
                        "s": word_start,
                        "e": word_end
                    })
                    segment_text_parts.append(word_text)
                    
            # Create segment if we have words
            if segment_words:
                segment_text = " ".join(segment_text_parts).strip()
                
                # Use actual word boundaries for segment timing
                actual_start = min(w["s"] for w in segment_words)
                actual_end = max(w["e"] for w in segment_words)
                
                aligned_segments.append({
                    "start": actual_start,
                    "end": actual_end,
                    "speaker": speaker_id,
                    "text": segment_text,
                    "words": segment_words,
                    "confidence": diar_segment.get("confidence", 1.0)
                })
                
        # Sort by start time
        aligned_segments.sort(key=lambda s: s["start"])
        
        return aligned_segments
        
    def _time_overlap(self, start1: float, end1: float, start2: float, end2: float) -> bool:
        """
        Check if two time intervals overlap.
        
        Args:
            start1, end1: First interval
            start2, end2: Second interval
            
        Returns:
            True if intervals overlap
        """
        return not (end1 <= start2 or end2 <= start1)
        
    def _merge_consecutive_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Merge consecutive segments from the same speaker with small gaps.
        
        Args:
            segments: List of aligned segments
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []
            
        merged = []
        current_segment = segments[0].copy()
        
        for next_segment in segments[1:]:
            # Check if we should merge with current segment
            same_speaker = current_segment["speaker"] == next_segment["speaker"]
            small_gap = (next_segment["start"] - current_segment["end"]) <= SEGMENT_MERGE_GAP_THRESHOLD
            
            if same_speaker and small_gap:
                # Merge segments
                current_segment["end"] = next_segment["end"]
                current_segment["text"] += " " + next_segment["text"]
                current_segment["words"].extend(next_segment["words"])
                
                # Update confidence (average)
                current_conf = current_segment.get("confidence", 1.0)
                next_conf = next_segment.get("confidence", 1.0)
                current_segment["confidence"] = (current_conf + next_conf) / 2
                
            else:
                # Start new segment
                merged.append(current_segment)
                current_segment = next_segment.copy()
                
        # Add the last segment
        merged.append(current_segment)
        
        logger.info(f"Merged {len(segments)} segments into {len(merged)} segments")
        return merged
        
    def _apply_speaker_labels(
        self, 
        segments: List[Dict], 
        speaker_mapping: Dict[int, str]
    ) -> List[Dict]:
        """
        Apply human-readable speaker labels to segments.
        
        Args:
            segments: List of segments with numeric speaker IDs
            speaker_mapping: Mapping from speaker ID to label
            
        Returns:
            List of segments with speaker labels
        """
        labeled_segments = []
        
        for segment in segments:
            labeled_segment = segment.copy()
            speaker_id = segment["speaker"]
            speaker_label = speaker_mapping.get(speaker_id, f"SPEAKER_{speaker_id}")
            
            labeled_segment["speaker"] = speaker_label
            labeled_segments.append(labeled_segment)
            
        return labeled_segments
        
    def save_srt(self, result: Dict, output_path: str):
        """
        Save the result as an SRT subtitle file.
        
        Args:
            result: Fused transcription result
            output_path: Path to save SRT file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result["segments"], 1):
                start_time = self._seconds_to_srt_time(segment["start"])
                end_time = self._seconds_to_srt_time(segment["end"])
                speaker = segment["speaker"]
                text = segment["text"]
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"[{speaker}] {text}\n")
                f.write("\n")
                
        logger.info(f"Saved SRT file: {output_path}")
        
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """
        Convert seconds to SRT time format (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            SRT formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
        
    def save_json(self, result: Dict, output_path: str):
        """
        Save the result as a JSON file.
        
        Args:
            result: Fused transcription result
            output_path: Path to save JSON file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=JSONL_INDENT, ensure_ascii=False)
            
        logger.info(f"Saved JSON file: {output_path}")
        
    def validate_result(self, result: Dict) -> Dict:
        """
        Validate the fused result for quality issues.
        
        Args:
            result: Fused transcription result
            
        Returns:
            Dict with validation warnings and stats
        """
        warnings = []
        stats = {}
        
        segments = result.get("segments", [])
        
        # Check for gaps between segments
        gaps = []
        for i in range(len(segments) - 1):
            gap = segments[i + 1]["start"] - segments[i]["end"]
            if gap > 1.0:  # Gap > 1 second
                gaps.append(gap)
                
        if gaps:
            warnings.append(f"Found {len(gaps)} gaps > 1s (max: {max(gaps):.1f}s)")
            
        # Check for very short segments
        short_segments = [s for s in segments if (s["end"] - s["start"]) < 0.5]
        if short_segments:
            warnings.append(f"Found {len(short_segments)} segments < 0.5s")
            
        # Check speaker distribution
        speaker_counts = {}
        for segment in segments:
            speaker = segment["speaker"]
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            
        # Check for speaker dominance
        if len(speaker_counts) > 1:
            max_count = max(speaker_counts.values())
            total_segments = len(segments)
            dominance = max_count / total_segments
            
            if dominance > 0.9:
                warnings.append(f"One speaker dominates {dominance*100:.1f}% of segments")
                
        stats = {
            "total_segments": len(segments),
            "speakers": list(speaker_counts.keys()),
            "speaker_segment_counts": speaker_counts,
            "total_duration": result.get("duration_sec", 0),
            "total_gaps": len(gaps),
            "avg_gap": sum(gaps) / len(gaps) if gaps else 0
        }
        
        return {
            "warnings": warnings,
            "stats": stats,
            "valid": len(warnings) == 0
        }