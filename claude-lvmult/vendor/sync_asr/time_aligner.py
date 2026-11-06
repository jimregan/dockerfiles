# Copyright (c) 2024, Jim O'Regan for Språkbanken Tal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Two-pass alignment of two chronologically-ordered timed sequences (e.g.
wav2vec output aligned against phonetic-model output over the same audio).

Pass 1 anchors elements whose start time and duration both match within
tolerance: high-confidence 1:1 correspondences.

Pass 2 covers everything between anchors by grouping elements that overlap
in time, preferring 1:1 pairs and only falling back to 1:N / N:1 groups
where the overlap is genuinely ambiguous.
"""
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..elements import TimedElement

# Match-method vocabulary for AlignedGroup.metadata["match_method"], chosen
# to read directly as a correspondence artifact's metadata (see corpus-build
# ADR 0008 and alignment/ctm_align.py's MATCH_METHOD constant) without
# translation at wrap time.
MATCH_METHOD_EXACT_TIMING = "exact_timing"
MATCH_METHOD_TIME_OVERLAP = "time_overlap"


@dataclass
class AlignedGroup:
    a_indices: List[int]
    b_indices: List[int]
    kind: str  # "exact", "overlap", "unmatched_a", "unmatched_b"
    # open bag for how this correspondence was established and any quality
    # flags -- e.g. {"match_method": MATCH_METHOD_TIME_OVERLAP,
    # "multi_token_span": True}. Empty for unmatched_a/unmatched_b: nothing
    # was established. Maps directly onto a corpus-build correspondence
    # artifact's `metadata` field.
    metadata: Dict[str, Any] = field(default_factory=dict)


def _is_tight_match(a: TimedElement, b: TimedElement, start_tolerance, duration_tolerance):
    return (abs(a.start_time - b.start_time) <= start_tolerance and
            abs(a.get_duration() - b.get_duration()) <= duration_tolerance)


def _find_anchors(a: List[TimedElement], b: List[TimedElement], start_tolerance, duration_tolerance):
    b_starts = [item.start_time for item in b]
    anchors = []
    last_j = -1
    for i, a_item in enumerate(a):
        lo = bisect_left(b_starts, a_item.start_time - start_tolerance, last_j + 1)
        candidates = []
        j = lo
        while j < len(b) and b[j].start_time <= a_item.start_time + start_tolerance:
            if _is_tight_match(a_item, b[j], start_tolerance, duration_tolerance):
                candidates.append(j)
            j += 1
        if candidates:
            best_j = min(
                candidates,
                key=lambda j: abs(a_item.start_time - b[j].start_time) + abs(a_item.get_duration() - b[j].get_duration()),
            )
            anchors.append((i, best_j))
            last_j = best_j
    return anchors


def _union_find(n):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    return find, union


def _components(a_slice, b_slice):
    na = len(a_slice)
    find, union = _union_find(na + len(b_slice))
    for ai, a_item in enumerate(a_slice):
        for bi, b_item in enumerate(b_slice):
            if a_item.overlap(b_item) > 0:
                union(ai, na + bi)

    groups = {}
    for ai in range(na):
        groups.setdefault(find(ai), {"a": [], "b": []})["a"].append(ai)
    for bi in range(len(b_slice)):
        groups.setdefault(find(na + bi), {"a": [], "b": []})["b"].append(bi)
    return sorted(groups.values(), key=lambda g: (g["a"][0] if g["a"] else g["b"][0]))


def _decompose_component(a_slice, b_slice, comp_a, comp_b):
    pairs = []
    for ai in comp_a:
        for bi in comp_b:
            ov = a_slice[ai].overlap(b_slice[bi])
            if ov > 0:
                pairs.append((ov, ai, bi))
    pairs.sort(key=lambda p: p[0], reverse=True)

    matched = []
    a_to_group = {}
    b_to_group = {}
    for _, ai, bi in pairs:
        if ai in a_to_group or bi in b_to_group:
            continue
        group = {"a": [ai], "b": [bi]}
        matched.append(group)
        a_to_group[ai] = group
        b_to_group[bi] = group

    def attach_leftover(index, own_side, own_to_group, other_slice, other_comp, other_to_group):
        best_group, best_overlap = None, 0
        own_item = a_slice[index] if own_side == "a" else b_slice[index]
        for other_index in other_comp:
            if other_index not in other_to_group:
                continue
            other_item = other_slice[other_index]
            ov = own_item.overlap(other_item)
            if ov > best_overlap:
                best_overlap = ov
                best_group = other_to_group[other_index]
        if best_group is None:
            best_group = {"a": [], "b": []}
            matched.append(best_group)
        best_group[own_side].append(index)
        own_to_group[index] = best_group

    for ai in comp_a:
        if ai not in a_to_group:
            attach_leftover(ai, "a", a_to_group, b_slice, comp_b, b_to_group)
    for bi in comp_b:
        if bi not in b_to_group:
            attach_leftover(bi, "b", b_to_group, a_slice, comp_a, a_to_group)

    groups = []
    for g in matched:
        g["a"].sort()
        g["b"].sort()
        groups.append(g)
    return groups


def _align_gap(a_slice, b_slice, a_offset, b_offset):
    if not a_slice and not b_slice:
        return []
    if not a_slice:
        return [AlignedGroup([], list(range(b_offset, b_offset + len(b_slice))), "unmatched_b")]
    if not b_slice:
        return [AlignedGroup(list(range(a_offset, a_offset + len(a_slice))), [], "unmatched_a")]

    groups = []
    for comp in _components(a_slice, b_slice):
        if not comp["a"]:
            groups.append(AlignedGroup([], [b_offset + i for i in comp["b"]], "unmatched_b"))
            continue
        if not comp["b"]:
            groups.append(AlignedGroup([a_offset + i for i in comp["a"]], [], "unmatched_a"))
            continue
        for g in _decompose_component(a_slice, b_slice, comp["a"], comp["b"]):
            metadata = {"match_method": MATCH_METHOD_TIME_OVERLAP}
            if len(g["a"]) > 1 or len(g["b"]) > 1:
                metadata["multi_token_span"] = True
            groups.append(AlignedGroup(
                [a_offset + i for i in g["a"]],
                [b_offset + i for i in g["b"]],
                "overlap",
                metadata,
            ))
    groups.sort(key=lambda g: (g.a_indices[0] if g.a_indices else a_offset + len(a_slice),
                                g.b_indices[0] if g.b_indices else b_offset + len(b_slice)))
    return groups


def align(
    a: List[TimedElement],
    b: List[TimedElement],
    start_tolerance: float = 0.02,
    duration_tolerance: float = 0.05,
) -> List[AlignedGroup]:
    """
    Align two chronologically-ordered sequences of TimedElement.

    `start_tolerance` and `duration_tolerance` are in whatever unit `a`
    and `b`'s start_time/end_time use (seconds or milliseconds); they
    control pass 1's exact-anchor matching only. Pass 2 falls back to
    pure time-overlap for everything anchors don't cover.
    """
    anchors = _find_anchors(a, b, start_tolerance, duration_tolerance)

    groups = []
    prev_i, prev_j = -1, -1
    for ai, bj in anchors:
        groups.extend(_align_gap(a[prev_i + 1:ai], b[prev_j + 1:bj], prev_i + 1, prev_j + 1))
        groups.append(AlignedGroup([ai], [bj], "exact", {"match_method": MATCH_METHOD_EXACT_TIMING}))
        prev_i, prev_j = ai, bj
    groups.extend(_align_gap(a[prev_i + 1:], b[prev_j + 1:], prev_i + 1, prev_j + 1))
    return groups
