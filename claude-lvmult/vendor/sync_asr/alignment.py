# Copyright (c) 2026, Jim O'Regan for Språkbanken Tal
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
AlignedSequence: a whole alignment pass (over timed or plain-text node
sequences) as a single, annotatable, comparable object -- and a pluggable
heuristic mechanism for bridging spans a base aligner left unresolved
with an accepted, named, non-destructive interpretation.

Reuses time_aligner.AlignedGroup as the unit of correspondence regardless
of domain (timing-based or text-based): `kind` is a free string, so a
text aligner can use edit-type vocabulary (cor/sub/ins/del/sil, matching
corpus-build's CTMEdit) while time_aligner.py keeps its own
(exact/overlap/unmatched_a/unmatched_b).
"""
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .elements import TimedElement
from .kaldi.align_ctm_ref import smith_waterman_alignment as _smith_waterman_alignment
from .riksdag.time_aligner import AlignedGroup

# Edit-kind vocabulary for text alignment, matching corpus-build's
# CTMEdit.edit_type (cor | sub | ins | del | sil) for consistency across
# the two codebases rather than inventing a parallel vocabulary.
KIND_CORRESPONDS = "cor"
KIND_SUBSTITUTE = "sub"
KIND_INSERT = "ins"
KIND_DELETE = "del"

TEXT_GAP_KINDS = {KIND_SUBSTITUTE, KIND_INSERT, KIND_DELETE}


def is_text_gap(group: AlignedGroup) -> bool:
    """The default `is_gap` predicate for align_text()'s output: any group
    that isn't a direct correspondence -- a substitution, insertion, or
    deletion left over from the base alignment. Exposed so callers doing
    a "later" refinement pass (new heuristics, more context available)
    can reuse it against a sequence align_text() already produced."""
    return group.kind in TEXT_GAP_KINDS


def _node_times(node) -> Optional[Tuple[float, float]]:
    if hasattr(node, "start_time") and hasattr(node, "end_time"):
        return node.start_time, node.end_time
    return None


def _node_text(node) -> str:
    return node.text if hasattr(node, "text") else str(node)


def _comparison_text(text: str, *, ignore_case: bool, ignore_punctuation: bool) -> str:
    if ignore_case:
        text = text.casefold()
    if ignore_punctuation:
        text = "".join(char for char in text
                        if not unicodedata.category(char).startswith("P"))
    return text


class AlignedSequence(TimedElement):
    """
    An ordered sequence of AlignedGroup correspondences between two node
    sequences `a` and `b`, produced by one alignment pass, carrying its
    own sequence-level annotations.

    Subclasses TimedElement so two AlignedSequences describing
    correspondences over the same underlying timeline -- e.g. a
    word-level ASR pass and a phonetic-level ASR pass reconciling the
    same reference span -- are directly comparable with the inherited
    overlap()/within()/contains()/has_overlap(): no separate comparison
    machinery needed. Span is computed from whichever side of each group
    actually carries timing (a reference-text side usually won't); a
    sequence with no timed nodes at all gets a zero-width span at time 0
    and isn't usefully time-comparable.
    """

    def __init__(
        self,
        a: Sequence,
        b: Sequence,
        groups: List[AlignedGroup],
        method: str,
        annotations: Optional[Dict[str, Any]] = None,
    ):
        self.a = a
        self.b = b
        self.groups = groups
        self.method = method
        start_time, end_time = self._span(a, b, groups)
        super().__init__(start_time, end_time, text="", annotations=annotations)

    @staticmethod
    def _span(a, b, groups) -> Tuple[float, float]:
        times: List[float] = []
        for g in groups:
            for i in g.a_indices:
                t = _node_times(a[i])
                if t is not None:
                    times.extend(t)
            for j in g.b_indices:
                t = _node_times(b[j])
                if t is not None:
                    times.extend(t)
        if not times:
            return 0, 0
        return min(times), max(times)


class AlignmentHeuristic(ABC):
    """
    A named, pluggable check that can explain an otherwise-unresolved
    span between two node sequences as an accepted variant (a
    paraphrase, a grammatical alternation, ...).

    Never alters the underlying nodes -- it only proposes additional
    AlignedGroup(s) over the same span, annotated with which heuristic
    produced them. bridge_gaps() never removes the span's original,
    unresolved group: which correspondence to trust is a query-time
    decision (see corpus-build ADR 0008), not one a heuristic gets to
    make unilaterally by deleting the alternative.
    """

    name: str

    @abstractmethod
    def match(self, a_span: Sequence, b_span: Sequence) -> Optional[List[AlignedGroup]]:
        """
        a_span, b_span: the actual node objects for one contiguous,
        currently-unresolved span (local, 0-based indices into these two
        lists -- not the parent sequence's).

        Return None if this heuristic doesn't explain this span.
        Otherwise, a list of AlignedGroup covering it, in local indices
        (bridge_gaps() rebases them onto the parent sequence) -- ideally
        a finer subsequence decomposition (e.g. word-for-word) rather
        than one opaque group, each carrying
        metadata={"heuristic": self.name, "accepted": True, ...}.
        """


class RepeatedTokenHeuristic(AlignmentHeuristic):
    """
    Recognises a span where every non-distributable token matches
    positionally between `a_span` and `b_span`, but some caller-named
    "distributable" token appears fewer times in `a_span` (e.g. once,
    citation-style) than in `b_span` (e.g. repeated at every item
    boundary). Worked example: ref "A, B och C" / hyp "A och B och C",
    with `distributable_tokens={"och", "eller"}` -- but the pattern
    itself isn't conjunction-specific: any token that can legitimately
    appear either once (eliding repeats) or repeated fits it, so the
    same class covers other reduced/expanded particles too, not just
    list conjunctions. `distributable_tokens` is caller-supplied --
    nothing is inferred from the data or hardcoded to any language.
    """

    def __init__(self, distributable_tokens: Iterable[str], name: str = "repeated_token_paraphrase"):
        self.name = name
        self._distributable_tokens = set(distributable_tokens)

    def _split_items(self, span: Sequence) -> Tuple[List[int], List[str]]:
        item_indices = [i for i, node in enumerate(span) if _node_text(node) not in self._distributable_tokens]
        items = [_node_text(span[i]) for i in item_indices]
        return item_indices, items

    def match(self, a_span: Sequence, b_span: Sequence) -> Optional[List[AlignedGroup]]:
        a_item_idx, a_items = self._split_items(a_span)
        b_item_idx, b_items = self._split_items(b_span)
        if len(a_items) < 2 or a_items != b_items:
            return None

        a_dist_idx = [i for i in range(len(a_span)) if i not in a_item_idx]
        b_dist_idx = [i for i in range(len(b_span)) if i not in b_item_idx]
        if len(b_dist_idx) <= len(a_dist_idx):
            return None  # nothing for this heuristic to explain

        groups = [
            AlignedGroup([ai], [bi], KIND_CORRESPONDS, {"heuristic": self.name, "accepted": True})
            for ai, bi in zip(a_item_idx, b_item_idx)
        ]

        if a_dist_idx and b_dist_idx:
            # a's single distributable token and b's last occurrence both
            # sit immediately before the final item -- direct correspondence.
            groups.append(AlignedGroup([a_dist_idx[-1]], [b_dist_idx[-1]], KIND_CORRESPONDS,
                                        {"heuristic": self.name, "accepted": True}))
            extra_b_dist = b_dist_idx[:-1]
        elif a_dist_idx:
            groups.append(AlignedGroup(a_dist_idx, [], KIND_DELETE,
                                        {"heuristic": self.name, "accepted": True}))
            extra_b_dist = b_dist_idx
        else:
            extra_b_dist = b_dist_idx

        if extra_b_dist:
            groups.append(AlignedGroup([], extra_b_dist, KIND_INSERT,
                                        {"heuristic": self.name, "accepted": True,
                                         "note": "distributable token repeated per item"}))

        groups.sort(key=lambda g: (g.a_indices[0] if g.a_indices else len(a_span),
                                    g.b_indices[0] if g.b_indices else len(b_span)))
        return groups


class WordPairDictionaryHeuristic(AlignmentHeuristic):
    """
    Generic dictionary-backed heuristic: accepts a mapping of known
    equivalent (a_phrase, b_phrase) tuples -- e.g. grammatical
    alternations such as preteritum vs har + supine -- and matches a
    span against it exactly. No language data is bundled here; callers
    supply their own `equivalences`.
    """

    def __init__(self, name: str, equivalences: Dict[Tuple[str, ...], Tuple[str, ...]]):
        self.name = name
        self._equivalences = equivalences

    def match(self, a_span: Sequence, b_span: Sequence) -> Optional[List[AlignedGroup]]:
        a_key = tuple(_node_text(n) for n in a_span)
        b_key = tuple(_node_text(n) for n in b_span)
        if self._equivalences.get(a_key) != b_key:
            return None
        return [AlignedGroup(
            list(range(len(a_span))), list(range(len(b_span))), KIND_CORRESPONDS,
            {"heuristic": self.name, "accepted": True},
        )]


class MergedTokenHeuristic(AlignmentHeuristic):
    """
    Recognises a span where a single hypothesis token's text equals the
    concatenation of the span's 2+ reference words with no separator --
    e.g. wav2vec recognising "Inläsningstjänst AB" as one word
    "INLÄSNINGSTJÄNSTAB" -- and splits that one token's time span across
    the reference words it represents, proportionally by character
    count (the obvious default with no finer-grained timing available at
    this stage). The split points are recorded as
    metadata["split_start_time"]/["split_end_time"] on each resulting
    group rather than invented TimedElement objects, since the parent
    AlignedSequence's `b` list still only has the one merged token --
    a caller building per-reference-word timing from these groups (see
    word_pronunciations.align_reference_timing) reads the split points
    instead of the referenced token's own full start/end.
    """

    name = "merged_token_split"

    def match(self, a_span: Sequence, b_span: Sequence) -> Optional[List[AlignedGroup]]:
        if len(a_span) < 2 or len(b_span) != 1:
            return None
        hyp = b_span[0]
        if not hasattr(hyp, "start_time") or not hasattr(hyp, "end_time"):
            return None
        ref_texts = [_node_text(n) for n in a_span]
        if "".join(ref_texts) != _node_text(hyp):
            return None
        total_chars = sum(len(t) for t in ref_texts)
        if total_chars == 0:
            return None

        span_duration = hyp.end_time - hyp.start_time
        groups = []
        cursor = hyp.start_time
        for i, text in enumerate(ref_texts):
            is_last = i == len(ref_texts) - 1
            end = hyp.end_time if is_last else cursor + span_duration * (len(text) / total_chars)
            groups.append(AlignedGroup(
                [i], [0], KIND_CORRESPONDS,
                {"heuristic": self.name, "accepted": True,
                 "split_start_time": cursor, "split_end_time": end},
            ))
            cursor = end
        return groups


def _gap_runs(groups: List[AlignedGroup], is_gap: Callable[[AlignedGroup], bool]):
    """Yield maximal runs of consecutive groups where is_gap(g) is True."""
    run: List[AlignedGroup] = []
    for g in groups:
        if is_gap(g):
            run.append(g)
        else:
            if run:
                yield run
            run = []
    if run:
        yield run


def _try_heuristics(sequence, a_indices, b_indices, heuristics):
    a_span = [sequence.a[i] for i in a_indices]
    b_span = [sequence.b[j] for j in b_indices]
    for heuristic in heuristics:
        bridge = heuristic.match(a_span, b_span)
        if bridge is None:
            continue
        return [
            AlignedGroup(
                a_indices=[a_indices[i] for i in bg.a_indices],
                b_indices=[b_indices[j] for j in bg.b_indices],
                kind=bg.kind,
                metadata=bg.metadata,
            )
            for bg in bridge
        ]
    return None


def bridge_gaps(
    sequence: AlignedSequence,
    heuristics: Iterable[AlignmentHeuristic],
    is_gap: Callable[[AlignedGroup], bool],
) -> AlignedSequence:
    """
    Try each heuristic against every maximal run of consecutive groups in
    `sequence` for which `is_gap(group)` is True -- first against the
    *whole run* combined (needed for aligners like align_smith_waterman
    that produce one group per del/ins/sub step rather than a block, so
    a pattern spanning several adjacent steps, e.g. a merged hypothesis
    token covering two reference words split across a del+sub pair,
    is visible as one span), then, if nothing matched the combined run,
    against each group in the run individually (align_text()'s existing,
    already-block-level "sub" gaps keep working exactly as before, since
    a run of length 1 skips straight to this fallback).

    The first heuristic that matches has its bridge groups appended to
    `sequence.groups` (index-rebased onto the parent sequence); the
    original gap group(s) are left untouched -- bridging is additive,
    both interpretations remain queryable.

    `is_gap` is caller-supplied because what counts as "unresolved" is
    domain-specific (time_aligner's own "unmatched_a"/"unmatched_b"
    kinds mean something quite different from a text aligner's "sub").
    Mutates and returns `sequence`.
    """
    new_groups = []
    heuristics = list(heuristics)
    for run in _gap_runs(sequence.groups, is_gap):
        bridged = None
        if len(run) > 1:
            a_indices = [i for g in run for i in g.a_indices]
            b_indices = [j for g in run for j in g.b_indices]
            bridged = _try_heuristics(sequence, a_indices, b_indices, heuristics)
        if bridged is not None:
            new_groups.extend(bridged)
            continue
        for g in run:
            bridged = _try_heuristics(sequence, g.a_indices, g.b_indices, heuristics)
            if bridged is not None:
                new_groups.extend(bridged)
    sequence.groups = sequence.groups + new_groups
    sequence.groups.sort(key=lambda g: (
        g.a_indices[0] if g.a_indices else float("inf"),
        g.b_indices[0] if g.b_indices else float("inf"),
    ))
    return sequence


MATCH_METHOD_TEXT_EQUAL = "text_equal"


def align_text(
    a: Sequence,
    b: Sequence,
    heuristics: Optional[Iterable[AlignmentHeuristic]] = None,
) -> AlignedSequence:
    """
    Align two plain node sequences (e.g. reference words vs an ASR
    hypothesis -- elements may be plain strings or richer objects with a
    `.text` attribute) by exact-match anchors (SequenceMatcher's "equal"
    blocks, decomposed to one AlignedGroup per position); everything
    else is left as an unresolved cor/sub/ins/del gap (see is_text_gap).

    If `heuristics` is given, they're applied immediately via
    bridge_gaps() before returning -- heuristics known "in advance." A
    caller can also call bridge_gaps() again later, with the same or
    different heuristics, once more context makes a further
    reinterpretation possible; both calls share the same non-destructive
    mechanism, only the timing differs.
    """
    a_text = [_node_text(n) for n in a]
    b_text = [_node_text(n) for n in b]
    matcher = SequenceMatcher(None, a_text, b_text, autojunk=False)

    groups: List[AlignedGroup] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                groups.append(AlignedGroup(
                    [i1 + offset], [j1 + offset], KIND_CORRESPONDS,
                    {"match_method": MATCH_METHOD_TEXT_EQUAL},
                ))
        elif tag == "delete":
            groups.append(AlignedGroup(list(range(i1, i2)), [], KIND_DELETE, {}))
        elif tag == "insert":
            groups.append(AlignedGroup([], list(range(j1, j2)), KIND_INSERT, {}))
        elif tag == "replace":
            groups.append(AlignedGroup(list(range(i1, i2)), list(range(j1, j2)), KIND_SUBSTITUTE, {}))

    sequence = AlignedSequence(a, b, groups, method="text_sequence_matcher")
    if heuristics:
        bridge_gaps(sequence, heuristics, is_gap=is_text_gap)
    return sequence


MATCH_METHOD_SMITH_WATERMAN = "smith_waterman_ctm_alignment"
_SW_EPS = "\x00<eps>\x00"  # sentinel; must not collide with a real token


def align_smith_waterman(
    ref: Sequence,
    hyp: Sequence,
    *,
    correct_score: int = 1,
    substitution_penalty: int = 1,
    deletion_penalty: int = 1,
    insertion_penalty: int = 1,
    align_full_hyp: bool = True,
    heuristics: Optional[Iterable[AlignmentHeuristic]] = None,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
) -> AlignedSequence:
    """
    Smith-Waterman alignment of `ref` against `hyp` -- the Kaldi-derived
    local aligner in kaldi/align_ctm_ref.py, unchanged (its core DP
    is shared, tested code; see test_align_ctm_ref.py) -- wrapped into
    an AlignedSequence/AlignedGroup so it composes with the rest of this
    module (bridge_gaps, comparability, annotations) instead of
    align_ctm_ref.py's own CLI's legacy flat CTM-edit-line format.

    ref, hyp: plain strings or richer objects with a `.text` attribute
    (e.g. TimedWord) -- hyp's timing, if any, flows through to the
    resulting AlignedSequence's span the same way align_text()'s does,
    since `hyp` itself (not just its text) becomes AlignedSequence.b.

    `align_full_hyp=True` (the default, matching align_ctm_ref.py's own
    default) tracks back from the end of `hyp`, giving the reference
    span that best matches the *entire* hypothesis -- appropriate when
    `hyp` is one complete utterance, as opposed to finding the best-
    matching reference subsequence for a hypothesis fragment.
    """
    ref_values = [_comparison_text(_node_text(n), ignore_case=ignore_case,
                                   ignore_punctuation=ignore_punctuation) for n in ref]
    hyp_values = [_comparison_text(_node_text(n), ignore_case=ignore_case,
                                   ignore_punctuation=ignore_punctuation) for n in hyp]
    ref_positions = [i for i, value in enumerate(ref_values) if value]
    hyp_positions = [i for i, value in enumerate(hyp_values) if value]
    ref_text = [ref_values[i] for i in ref_positions]
    hyp_text = [hyp_values[i] for i in hyp_positions]

    def similarity(a, b):
        return correct_score if a == b else -substitution_penalty

    raw, _score = _smith_waterman_alignment(
        ref_text, hyp_text, similarity_score_function=similarity,
        del_score=-deletion_penalty, ins_score=-insertion_penalty,
        eps_symbol=_SW_EPS, align_full_hyp=align_full_hyp,
    )

    groups: List[AlignedGroup] = []
    for ref_word, hyp_word, _prev_ref_i, _prev_hyp_i, cur_ref_i, cur_hyp_i in raw:
        a_indices = [] if ref_word == _SW_EPS else [ref_positions[cur_ref_i - 1]]
        b_indices = [] if hyp_word == _SW_EPS else [hyp_positions[cur_hyp_i - 1]]
        if a_indices and b_indices:
            kind = KIND_CORRESPONDS if ref_word == hyp_word else KIND_SUBSTITUTE
        elif b_indices:
            kind = KIND_INSERT
        else:
            kind = KIND_DELETE
        groups.append(AlignedGroup(a_indices, b_indices, kind,
                                    {"match_method": MATCH_METHOD_SMITH_WATERMAN}))

    sequence = AlignedSequence(list(ref), list(hyp), groups, method=MATCH_METHOD_SMITH_WATERMAN)
    if heuristics:
        bridge_gaps(sequence, heuristics, is_gap=is_text_gap)
    return sequence
