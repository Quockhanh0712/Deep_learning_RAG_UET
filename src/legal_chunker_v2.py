"""
Legal Document Chunker v2
Intelligent chunking for Vietnamese legal documents with proper hierarchy handling
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class LegalChunk:
    """Represents a chunk of legal document with metadata"""
    chunk_id: str
    text: str
    van_ban_id: str
    ten_van_ban: str
    loai_van_ban: str
    co_quan: str
    chuong: str
    ten_chuong: str
    dieu_so: str
    tieu_de_dieu: str
    chunk_index: int
    total_chunks: int
    char_start: int
    char_end: int
    hierarchy: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'content': self.text,  # Alias for compatibility
            'van_ban_id': self.van_ban_id,
            'ten_van_ban': self.ten_van_ban,
            'loai_van_ban': self.loai_van_ban,
            'co_quan': self.co_quan,
            'chuong': self.chuong,
            'ten_chuong': self.ten_chuong,
            'dieu_so': self.dieu_so,
            'tieu_de_dieu': self.tieu_de_dieu,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,
            'char_start': self.char_start,
            'char_end': self.char_end,
            'hierarchy': self.hierarchy
        }


class LegalChunkerV2:
    """
    Legal-aware text chunker that respects Vietnamese legal document structure
    
    Hierarchy: Chương -> Điều -> Khoản -> Điểm
    
    Features:
    - Respects legal structure boundaries (Điều, Khoản, Điểm)
    - Maintains context with overlap
    - Generates unique chunk IDs
    - Preserves metadata for retrieval
    """
    
    # Patterns for Vietnamese legal structure
    DIEU_PATTERN = re.compile(r'^Điều\s+(\d+)\s*[.:]?\s*(.*)$', re.MULTILINE | re.IGNORECASE)
    KHOAN_PATTERN = re.compile(r'^(\d+)\.\s+', re.MULTILINE)
    DIEM_PATTERN = re.compile(r'^([a-zđ])\)\s+', re.MULTILINE | re.IGNORECASE)
    CHUONG_PATTERN = re.compile(r'^Chương\s+([IVXLCDM]+|\d+)\s*[.:]?\s*(.*)$', re.MULTILINE | re.IGNORECASE)
    
    def __init__(
        self,
        min_chars: int = 1200,
        max_chars: int = 2800,
        overlap_chars: int = 100
    ):
        """
        Initialize the legal chunker
        
        Args:
            min_chars: Minimum characters per chunk (default 1200)
            max_chars: Maximum characters per chunk (default 2800)
            overlap_chars: Overlap between chunks for context (default 100)
        """
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        logger.debug(f"[CHUNKER] Initialized: min={min_chars}, max={max_chars}, overlap={overlap_chars}")
    
    def _generate_chunk_id(self, van_ban_id: str, dieu_so: str, chunk_index: int, text: str) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{van_ban_id}_D{dieu_so}_C{chunk_index}_{content_hash}"
    
    def _find_split_point(self, text: str, start: int, end: int) -> int:
        """
        Find the best point to split text between start and end
        Priority: Điều > Khoản > Điểm > Sentence > Word
        """
        search_text = text[start:end]
        
        # Try to find a Điều boundary
        dieu_match = list(self.DIEU_PATTERN.finditer(search_text))
        if dieu_match:
            return start + dieu_match[-1].start()
        
        # Try to find a Khoản boundary
        khoan_match = list(self.KHOAN_PATTERN.finditer(search_text))
        if khoan_match:
            return start + khoan_match[-1].start()
        
        # Try to find a Điểm boundary
        diem_match = list(self.DIEM_PATTERN.finditer(search_text))
        if diem_match:
            return start + diem_match[-1].start()
        
        # Try to find sentence boundary (., ;, :)
        for i in range(len(search_text) - 1, -1, -1):
            if search_text[i] in '.;:\n':
                return start + i + 1
        
        # Fall back to word boundary
        for i in range(len(search_text) - 1, -1, -1):
            if search_text[i] == ' ':
                return start + i + 1
        
        return end
    
    def _extract_hierarchy(self, text: str) -> Dict:
        """Extract legal hierarchy from text"""
        hierarchy = {}
        
        # Find Chương
        chuong_match = self.CHUONG_PATTERN.search(text)
        if chuong_match:
            hierarchy['chuong'] = chuong_match.group(1)
            hierarchy['ten_chuong'] = chuong_match.group(2).strip()
        
        # Find Điều
        dieu_match = self.DIEU_PATTERN.search(text)
        if dieu_match:
            hierarchy['dieu'] = dieu_match.group(1)
            hierarchy['tieu_de_dieu'] = dieu_match.group(2).strip()
        
        # Find Khoản (first one)
        khoan_match = self.KHOAN_PATTERN.search(text)
        if khoan_match:
            hierarchy['khoan'] = khoan_match.group(1)
        
        # Find Điểm (first one)
        diem_match = self.DIEM_PATTERN.search(text)
        if diem_match:
            hierarchy['diem'] = diem_match.group(1)
        
        return hierarchy
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[LegalChunk]:
        """
        Chunk plain text with optional metadata
        
        Args:
            text: Text to chunk
            metadata: Optional metadata dict
            
        Returns:
            List of LegalChunk objects
        """
        if metadata is None:
            metadata = {
                'van_ban_id': 'UNKNOWN',
                'ten_van_ban': '',
                'loai_van_ban': '',
                'co_quan': '',
                'chuong': '',
                'ten_chuong': '',
                'dieu_so': '0',
                'tieu_de_dieu': ''
            }
        
        pseudo_row = {
            'van_ban_id': metadata.get('van_ban_id', 'UNKNOWN'),
            'ten_van_ban': metadata.get('ten_van_ban', ''),
            'loai_van_ban': metadata.get('loai_van_ban', ''),
            'co_quan': metadata.get('co_quan', ''),
            'chuong': metadata.get('chuong', ''),
            'ten_chuong': metadata.get('ten_chuong', ''),
            'dieu_so': metadata.get('dieu_so', '0'),
            'tieu_de_dieu': metadata.get('tieu_de_dieu', ''),
            'clean_text': text,
            'noi_dung': text
        }
        
        return self.chunk_article(pseudo_row, 0)
    
    def chunk_article(self, row: Dict, article_idx: int) -> List[LegalChunk]:
        """
        Chunk a legal article (Điều) with smart boundary detection
        
        Args:
            row: Dictionary containing article data with keys:
                - van_ban_id, ten_van_ban, loai_van_ban, co_quan
                - chuong, ten_chuong, dieu_so, tieu_de_dieu
                - clean_text or noi_dung
            article_idx: Index of the article (for ordering)
            
        Returns:
            List of LegalChunk objects
        """
        # Extract text
        text = row.get('clean_text') or row.get('noi_dung', '')
        if not text or not text.strip():
            return []
        
        text = text.strip()
        text_len = len(text)
        
        # If text is small enough, return as single chunk
        if text_len <= self.max_chars:
            chunk = LegalChunk(
                chunk_id=self._generate_chunk_id(
                    row.get('van_ban_id', 'UNK'),
                    row.get('dieu_so', '0'),
                    0,
                    text
                ),
                text=text,
                van_ban_id=row.get('van_ban_id', ''),
                ten_van_ban=row.get('ten_van_ban', ''),
                loai_van_ban=row.get('loai_van_ban', ''),
                co_quan=row.get('co_quan', ''),
                chuong=row.get('chuong', ''),
                ten_chuong=row.get('ten_chuong', ''),
                dieu_so=row.get('dieu_so', ''),
                tieu_de_dieu=row.get('tieu_de_dieu', ''),
                chunk_index=0,
                total_chunks=1,
                char_start=0,
                char_end=text_len,
                hierarchy=self._extract_hierarchy(text)
            )
            return [chunk]
        
        # Split into multiple chunks
        chunks = []
        current_pos = 0
        chunk_index = 0
        
        while current_pos < text_len:
            # Calculate chunk boundaries
            chunk_start = current_pos
            ideal_end = min(current_pos + self.max_chars, text_len)
            
            if ideal_end >= text_len:
                # Last chunk - take everything remaining
                chunk_end = text_len
            else:
                # Find best split point
                search_start = max(current_pos + self.min_chars, current_pos)
                chunk_end = self._find_split_point(text, search_start, ideal_end)
            
            # Extract chunk text
            chunk_text = text[chunk_start:chunk_end].strip()
            
            if chunk_text:
                chunk = LegalChunk(
                    chunk_id=self._generate_chunk_id(
                        row.get('van_ban_id', 'UNK'),
                        row.get('dieu_so', '0'),
                        chunk_index,
                        chunk_text
                    ),
                    text=chunk_text,
                    van_ban_id=row.get('van_ban_id', ''),
                    ten_van_ban=row.get('ten_van_ban', ''),
                    loai_van_ban=row.get('loai_van_ban', ''),
                    co_quan=row.get('co_quan', ''),
                    chuong=row.get('chuong', ''),
                    ten_chuong=row.get('ten_chuong', ''),
                    dieu_so=row.get('dieu_so', ''),
                    tieu_de_dieu=row.get('tieu_de_dieu', ''),
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    char_start=chunk_start,
                    char_end=chunk_end,
                    hierarchy=self._extract_hierarchy(chunk_text)
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move position with overlap
            current_pos = max(chunk_end - self.overlap_chars, chunk_end)
            
            # Safety: prevent infinite loop
            if current_pos <= chunk_start:
                current_pos = chunk_end
        
        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        return chunks


def get_legal_chunker(
    min_chars: int = 1200,
    max_chars: int = 2800,
    overlap_chars: int = 100
) -> LegalChunkerV2:
    """Factory function to create a legal chunker"""
    return LegalChunkerV2(
        min_chars=min_chars,
        max_chars=max_chars,
        overlap_chars=overlap_chars
    )
