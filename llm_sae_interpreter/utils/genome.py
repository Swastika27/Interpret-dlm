"""
utils/genome.py — Fetch DNA sequences from a reference FASTA using pyfaidx.

Handles chromosome name normalisation (chr1 vs 1), bounds clamping, and
optional flanking context expansion used in Step 6 reclustering.
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path

try:
    from pyfaidx import Fasta, FetchError
    PYFAIDX_AVAILABLE = True
except ImportError:
    PYFAIDX_AVAILABLE = False
    print("WARNING: pyfaidx not installed. Run: pip install pyfaidx")


@lru_cache(maxsize=1)
def _load_genome(fasta_path: str) -> "Fasta":
    """Load (and cache) the reference genome FASTA. Called once per process."""
    if not PYFAIDX_AVAILABLE:
        raise RuntimeError("pyfaidx is required: pip install pyfaidx")
    path = Path(fasta_path)
    if not path.exists():
        raise FileNotFoundError(f"Genome FASTA not found: {path}")
    return Fasta(str(path), sequence_always_upper=True)


def _normalise_chrom(genome: "Fasta", chrom: str) -> str:
    """Try both 'chr1' and '1' style chromosome names."""
    if chrom in genome:
        return chrom
    alt = chrom.lstrip("chr") if chrom.startswith("chr") else f"chr{chrom}"
    if alt in genome:
        return alt
    raise KeyError(f"Chromosome '{chrom}' not found in FASTA (tried '{alt}' too)")


def fetch_sequence(
    fasta_path: str,
    chrom: str,
    start: int,
    end: int,
    extra_flank: int = 0,
) -> tuple[str, int, int]:
    """
    Fetch a DNA sequence window from the reference genome.

    Parameters
    ----------
    fasta_path : path to hg38.fa
    chrom      : e.g. 'chr21'
    start      : 0-based start coordinate (BED-style)
    end        : 0-based exclusive end coordinate
    extra_flank: additional bp to add on each side (used in Step 6)

    Returns
    -------
    sequence   : uppercase DNA string (length = end - start + 2*extra_flank,
                 possibly shorter at chromosome boundaries)
    actual_start : actual 0-based start after clamping
    actual_end   : actual 0-based end after clamping
    """
    genome = _load_genome(fasta_path)
    chrom  = _normalise_chrom(genome, chrom)
    chrom_len = len(genome[chrom])

    actual_start = max(0, start - extra_flank)
    actual_end   = min(chrom_len, end + extra_flank)

    try:
        # pyfaidx uses 0-based half-open intervals when called with [start:end]
        seq = str(genome[chrom][actual_start:actual_end])
    except (FetchError, Exception) as exc:
        raise RuntimeError(
            f"Failed to fetch {chrom}:{actual_start}-{actual_end}: {exc}"
        ) from exc

    return seq.upper(), actual_start, actual_end


def highlight_position(
    sequence: str,
    tok_pos: int,
    seq_start: int,
    window_start: int,
) -> str:
    """
    Wrap the activating nucleotide in square brackets.

    Parameters
    ----------
    sequence     : full fetched sequence string
    tok_pos      : token position within the ORIGINAL sequence (from CSV)
    seq_start    : 0-based genomic start of the fetched sequence
    window_start : 0-based genomic start of the original 512 bp window

    Returns the sequence string with [X] at the activating position.
    """
    # Convert tok_pos (offset within original window) to offset within fetched seq
    genomic_pos  = window_start + tok_pos
    offset       = genomic_pos - seq_start

    if offset < 0 or offset >= len(sequence):
        # Position outside fetched window — return without highlighting
        return sequence

    return sequence[:offset] + f"[{sequence[offset]}]" + sequence[offset + 1:]


def truncate_for_display(sequence: str, max_chars: int = 200, center_on: str = "[") -> str:
    """
    Truncate a long sequence for display in prompts, keeping the highlighted
    position roughly centred.

    Parameters
    ----------
    sequence  : sequence string (may contain [X] highlight)
    max_chars : maximum display length including the [X] marker
    center_on : substring to center on (default '[' = the highlight bracket)
    """
    if len(sequence) <= max_chars:
        return sequence

    bracket_pos = sequence.find(center_on)
    if bracket_pos == -1:
        # No highlight — just take from start
        return sequence[:max_chars] + "…"

    half = max_chars // 2
    start = max(0, bracket_pos - half)
    end   = min(len(sequence), start + max_chars)

    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(sequence) else ""
    return prefix + sequence[start:end] + suffix
