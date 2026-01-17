"""arXiv paper downloading and management."""

import re
import tarfile
import gzip
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import httpx


class ArxivError(Exception):
    """Base exception for arXiv-related errors."""
    pass


def parse_arxiv_id(url_or_id: str) -> Tuple[str, Optional[str]]:
    """
    Parse arXiv URL or ID into normalized ID and optional version.
    
    Args:
        url_or_id: arXiv URL or ID (e.g., "2107.03374", "2107.03374v2", 
                   "https://arxiv.org/abs/2107.03374", "hep-th/9901001")
    
    Returns:
        Tuple of (arxiv_id, version) where version is None if not specified
        
    Examples:
        >>> parse_arxiv_id("2107.03374")
        ('2107.03374', None)
        >>> parse_arxiv_id("2107.03374v2")
        ('2107.03374', 'v2')
        >>> parse_arxiv_id("https://arxiv.org/abs/2107.03374v1")
        ('2107.03374', 'v1')
        >>> parse_arxiv_id("hep-th/9901001")
        ('hep-th/9901001', None)
    """
    # Extract from URL if needed
    if url_or_id.startswith("http://") or url_or_id.startswith("https://"):
        parsed = urlparse(url_or_id)
        if "arxiv.org" not in parsed.netloc:
            raise ArxivError(f"Not an arXiv URL: {url_or_id}")
        
        # Extract ID from path (/abs/..., /pdf/..., etc.)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) < 2:
            raise ArxivError(f"Cannot extract arXiv ID from URL: {url_or_id}")
        
        # For old-style IDs like /abs/hep-th/9901001, we need to join the last two parts
        # For new-style IDs like /abs/2107.03374, we just take the last part
        if len(path_parts) >= 3 and re.match(r"^[a-z\-]+$", path_parts[-2]):
            url_or_id = "/".join(path_parts[-2:])
        else:
            url_or_id = path_parts[-1]
        
        # Remove .pdf extension if present
        url_or_id = re.sub(r"\.pdf$", "", url_or_id)
    
    # Strip arXiv. or arXiv: prefix if present
    if url_or_id.startswith("arXiv.") or url_or_id.startswith("arXiv:"):
        url_or_id = url_or_id[6:]
    
    # Now parse the ID itself
    # New style: YYMM.NNNNN[vN]
    match = re.match(r"^(\d{4}\.\d{4,5})(v\d+)?$", url_or_id)
    if match:
        return match.group(1), match.group(2)
    
    # Old style: archive/YYMMNNN[vN]
    match = re.match(r"^([a-z\-]+/\d{7})(v\d+)?$", url_or_id)
    if match:
        return match.group(1), match.group(2)
    
    raise ArxivError(f"Invalid arXiv ID format: {url_or_id}")


def download_arxiv_source(
    arxiv_id: str,
    version: Optional[str] = None,
    output_dir: Path = Path("."),
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> Path:
    """
    Download arXiv source files.
    
    Args:
        arxiv_id: Normalized arXiv ID (e.g., "2107.03374" or "hep-th/9901001")
        version: Optional version string (e.g., "v2")
        output_dir: Directory to extract source files to
        max_retries: Number of retries on failure
        backoff_factor: Exponential backoff multiplier
        
    Returns:
        Path to the directory containing extracted source files
        
    Raises:
        ArxivError: If download or extraction fails
    """
    # Construct download URL
    id_with_version = f"{arxiv_id}{version}" if version else arxiv_id
    url = f"https://arxiv.org/src/{id_with_version}"
    
    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download with retries and backoff
    headers = {"User-Agent": "PaperVibe/0.1.0 (Research tool; mailto:user@example.com)"}
    
    for attempt in range(max_retries):
        try:
            with httpx.Client(headers=headers, follow_redirects=True, timeout=30.0) as client:
                response = client.get(url)
                response.raise_for_status()
                
                # Determine file type from content-type or try both
                content = response.content
                
                # Try as tar.gz first
                try:
                    temp_tar = output_dir / "temp_source.tar.gz"
                    temp_tar.write_bytes(content)
                    
                    with tarfile.open(temp_tar, "r:gz") as tar:
                        tar.extractall(path=output_dir, filter='data')
                    
                    temp_tar.unlink()
                    return output_dir
                    
                except (tarfile.TarError, gzip.BadGzipFile):
                    # Try as single .gz file
                    try:
                        if temp_tar.exists():
                            temp_tar.unlink()
                        
                        temp_gz = output_dir / "temp_source.gz"
                        temp_gz.write_bytes(content)
                        
                        # Extract single file
                        with gzip.open(temp_gz, "rb") as gz:
                            # Guess output filename (usually .tex)
                            output_file = output_dir / f"{arxiv_id.replace('/', '_')}.tex"
                            output_file.write_bytes(gz.read())
                        
                        temp_gz.unlink()
                        return output_dir
                        
                    except gzip.BadGzipFile:
                        # Might be uncompressed single file
                        output_file = output_dir / f"{arxiv_id.replace('/', '_')}.tex"
                        output_file.write_bytes(content)
                        return output_dir
                        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ArxivError(f"arXiv source not found: {id_with_version}")
            if attempt < max_retries - 1:
                time.sleep(backoff_factor ** attempt)
                continue
            raise ArxivError(f"HTTP error downloading arXiv source: {e}")
            
        except httpx.RequestError as e:
            if attempt < max_retries - 1:
                time.sleep(backoff_factor ** attempt)
                continue
            raise ArxivError(f"Network error downloading arXiv source: {e}")
    
    raise ArxivError(f"Failed to download arXiv source after {max_retries} attempts")
