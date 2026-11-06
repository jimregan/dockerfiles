"""N-gram contiguous region matching (kblabb approach)."""

import math

import numpy as np


def get_ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    """Generate n-grams from a list of words."""
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def ngram_match_vector(
    ref_ngrams: list[tuple[str, ...]], hyp_ngram_set: set[tuple[str, ...]]
) -> np.ndarray:
    """Boolean array: True where the ref n-gram appears in the hyp set."""
    return np.array(
        [ng in hyp_ngram_set for ng in ref_ngrams], dtype=np.float64
    )


def weighted_ngram_score(
    ref_words: list[str], hyp_words: list[str], max_n: int = 6
) -> np.ndarray:
    """Compute a weighted n-gram match score array over the reference words.

    For each n from 1..max_n, build n-grams of both texts, compute a boolean
    match vector, convolve with a kernel of size n+2 weighted by sqrt(log(n+1)),
    then stack and sum all layers, dividing by 3.
    """
    ref_len = len(ref_words)
    if ref_len == 0:
        return np.array([])

    layers = []
    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(ref_words, n)
        if not ref_ngrams:
            continue
        hyp_ngram_set = set(get_ngrams(hyp_words, n))
        match_vec = ngram_match_vector(ref_ngrams, hyp_ngram_set)

        kernel_size = n + 2
        weight = math.sqrt(math.log(n + 1))
        kernel = np.ones(kernel_size) * weight / kernel_size

        convolved = np.convolve(match_vec, kernel, mode="same")

        # Pad to ref_len so all layers have equal length
        padded = np.zeros(ref_len)
        offset = (n - 1) // 2
        padded[offset : offset + len(convolved)] = convolved
        layers.append(padded)

    if not layers:
        return np.zeros(ref_len)

    stacked = np.stack(layers)
    return stacked.sum(axis=0) / 3.0


def find_contiguous_regions(
    score_array: np.ndarray, threshold: float = 1.3
) -> list[tuple[int, int]]:
    """Find contiguous regions where score >= threshold.

    Returns list of (start, end) index pairs (end is exclusive).
    """
    above = score_array >= threshold
    regions = []
    start = None
    for i, val in enumerate(above):
        if val and start is None:
            start = i
        elif not val and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(above)))
    return regions


def filter_and_join_regions(
    regions: list[tuple[int, int]],
    min_length: int = 8,
    max_gap: int = 30,
) -> list[tuple[int, int]]:
    """Filter short regions and bridge small gaps between remaining ones."""
    # Filter by minimum length
    filtered = [(s, e) for s, e in regions if (e - s) >= min_length]
    if not filtered:
        return filtered

    # Bridge gaps
    merged = [filtered[0]]
    for s, e in filtered[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e <= max_gap:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))
    return merged


def contiguous_ngram_match(
    ref_words: list[str], hyp_words: list[str]
) -> tuple[int, int] | None:
    """Top-level n-gram match: return (start, end) word indices in ref, or None."""
    scores = weighted_ngram_score(ref_words, hyp_words)
    if len(scores) == 0:
        return None

    regions = find_contiguous_regions(scores)
    regions = filter_and_join_regions(regions)

    if not regions:
        return None

    # Return the longest region
    best = max(regions, key=lambda r: r[1] - r[0])
    return best
