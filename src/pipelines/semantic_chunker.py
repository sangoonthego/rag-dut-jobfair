import html
import re
from typing import List, Dict, Any

class JobDescriptionChunker:
    """
    Semantic chunker for Job Descriptions prioritizing logical breaks,
    metadata injection, and preventing orphaned chunks.
    """
    def __init__(self, target_size: int = 400, overlap: float = 0.15, min_chunk_size: int = 50):
        self.target_size = target_size
        self.overlap_size = int(target_size * overlap)
        self.min_chunk_size = min_chunk_size

    def normalize_text(self, text: str) -> str:
        """
        Convert HTML entities to plain text and clean up whitespaces while
        preserving structural line breaks.
        """
        # Convert HTML entities
        text = html.unescape(text)
        
        # Replace common unicode bullet points if needed, though keeping them is fine
        text = text.replace("▪", "-").replace("•", "-").replace("·", "-").replace("※", "-").replace("►", "-")
        
        # Strip redundant whitespaces but preserve single/double newlines
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n[ \t]+\n', '\n\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    def split_into_segments(self, text: str) -> List[str]:
        """
        Split text using semantic boundaries: double newlines, bullet points.
        """
        # Split by double newlines (paragraphs/sections)
        blocks = re.split(r'\n\n+', text)
        
        segments = []
        for block in blocks:
            # Check if block has multiple bullet points, split them carefully
            if re.search(r'\n\s*[-*]\s+', block):
                # Split along the bullets, keeping the bullet character
                bullet_lines = re.split(r'(?=\n\s*[-*]\s+)', block)
                for line in bullet_lines:
                    clean_line = line.strip()
                    if clean_line:
                        segments.append(clean_line)
            else:
                clean_block = block.strip()
                if clean_block:
                    segments.append(clean_block)
                    
        return segments

    def inject_metadata(self, chunk_text: str, metadata: Dict[str, Any], current_section: str = "General") -> str:
        """
        Inject context header derived from metadata.
        """
        company_name = metadata.get("company_name", "Unknown Company")
        job_title = metadata.get("job_title", "Unknown Position")
        
        header = f"[Company: {company_name}] | [Position: {job_title}] | [Section: {current_section}]\n"
        return header + chunk_text

    def estimate_tokens(self, text: str) -> int:
        """
        Rough heuristic for token count (approx 4 chars per token for English/Vietnamese).
        For production, use a proper tokenizer (like tiktoken or Transformers tokenizer).
        """
        return len(text) // 4

    def chunk_job_description(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """
        Execute the chunking strategy.
        """
        normalized_text = self.normalize_text(text)
        segments = self.split_into_segments(normalized_text)
        
        chunks = []
        current_chunk_segments = []
        current_chunk_tokens = 0
        current_section = "General"
        
        # Simple heuristic to detect section headers (all caps or trailing colon)
        def detect_section(segment: str) -> str:
            lines = segment.split('\n')
            first_line = lines[0].strip()
            if len(first_line) < 50 and (first_line.isupper() or first_line.endswith(':')):
                return first_line.rstrip(':')
            return current_section

        for segment in segments:
            # Update current section if the segment looks like a header
            new_section = detect_section(segment)
            if new_section != current_section:
                current_section = new_section
                
            segment_tokens = self.estimate_tokens(segment)
            
            # If a single segment is too large, we might need a sentence splitter
            # For simplicity in this logical definition, we append it directly
            # but ideally, we should split it further by sentences.
            
            if current_chunk_tokens + segment_tokens > self.target_size and current_chunk_segments:
                # Merge current segments into a chunk
                joined_chunk = "\n".join(current_chunk_segments)
                chunks.append(self.inject_metadata(joined_chunk, metadata, current_section))
                
                # Overlap logic (keep the last segment or two if it fits in overlap_size)
                overlap_segments = []
                overlap_tokens = 0
                for s in reversed(current_chunk_segments):
                    s_tokens = self.estimate_tokens(s)
                    if overlap_tokens + s_tokens <= self.overlap_size:
                        overlap_segments.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk_segments = overlap_segments
                current_chunk_segments.append(segment)
                current_chunk_tokens = overlap_tokens + segment_tokens
            else:
                current_chunk_segments.append(segment)
                current_chunk_tokens += segment_tokens
                
        # Handle the remaining segments
        if current_chunk_segments:
            joined_chunk = "\n".join(current_chunk_segments)
            
            # Prevent orphaned chunks: if the remainder is too small, append to the last chunk
            if chunks and self.estimate_tokens(joined_chunk) < self.min_chunk_size:
                # Need to strip the old headers off the last chunk before merging, or just append the text
                last_chunk_lines = chunks[-1].split('\n')
                # Header is the first line
                last_chunk_body = "\n".join(last_chunk_lines[1:])
                merged_body = last_chunk_body + "\n" + joined_chunk
                
                # Re-inject metadata with the current section (or keep the old one)
                chunks[-1] = self.inject_metadata(merged_body, metadata, current_section)
            else:
                chunks.append(self.inject_metadata(joined_chunk, metadata, current_section))
                
        return chunks
