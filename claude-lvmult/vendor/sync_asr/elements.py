# Copyright (c) 2022, Jim O'Regan for Språkbanken Tal
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
from typing import Any, Dict, List, Optional


class TimedElement():
    def __init__(self, start_time=0, end_time=0, text="", annotations: Optional[Dict[str, Any]] = None):
        self.start_time = start_time
        self.end_time = end_time
        self.text = text
        self.duration = end_time - start_time
        # open bag for whatever a transformation/alignment pass wants to
        # record about this node (match method, quality flags, etc.) --
        # mirrors the open `metadata` dict a corpus-build artifact carries.
        self.annotations: Dict[str, Any] = annotations if annotations is not None else {}

    def __str__(self) -> str:
        return f"[{self.start_time},{self.end_time}] {self.text}"

    def _is_valid_comparison(self, other):
        return (hasattr(other, "start_time") and
                hasattr(other, "end_time") and
                hasattr(other, "text"))

    def get_duration(self):
        if "duration" in self.__dict__:
            return self.duration
        else:
            return self.end_time - self.start_time

    def within(self, other, tolerance=0):
        """
        Whether self falls inside other's span. `tolerance` loosens both
        boundaries by that amount (same unit as start_time/end_time),
        for sources whose timing is approximate or a known conservative
        bound rather than exact (e.g. Whisper output, Riksdag speech API
        windows) -- controlled explicitly by the caller per comparison,
        not inferred from any per-node flag.
        """
        if not self._is_valid_comparison(other):
            return NotImplemented
        return (self.end_time <= other.end_time + tolerance and
                self.start_time >= other.start_time - tolerance)

    def contains(self, other, tolerance=0):
        """Whether `other` is entirely within self -- the inverse of within()."""
        if not self._is_valid_comparison(other):
            return NotImplemented
        return other.within(self, tolerance=tolerance)

    def __gt__(self, other):
        """Strictly greater than: self completely contains other"""
        return (self.start_time < other.start_time and self.end_time > other.end_time)

    def __lt__(self, other):
        """Strictly less than: other completely contains self"""
        return (self.start_time > other.start_time and self.end_time < other.end_time)

    def has_overlap(self, other, tolerance=0):
        if not hasattr(other, "duration"):
            return NotImplemented
        if not self._is_valid_comparison(other):
            return NotImplemented
        return self.start_time < other.end_time + tolerance and other.start_time < self.end_time + tolerance

    def contained_duration(self, other):
        # FIXME
        if not hasattr(other, "duration"):
            return NotImplemented

    def overlap(self, other, tolerance=0):
        if not self._is_valid_comparison(other):
            return NotImplemented
        self_start, self_end = self.start_time - tolerance, self.end_time + tolerance
        other_start, other_end = other.start_time - tolerance, other.end_time + tolerance
        if self_start < other_start:
            min1, min2 = self_start, other_start
        else:
            min2, min1 = self_start, other_start
        if self_end > other_end:
            max1, max2 = self_end, other_end
        else:
            max2, max1 = self_end, other_end
        return max(0, min(max1, max2) - max(min1, min2))

    def pct_overlap(self, other, tolerance=0):
        """Percentage of how much of this is contained by other"""
        if not hasattr(other, "duration"):
            return NotImplemented
        if not self.has_overlap(other, tolerance=tolerance):
            return 0.0
        if self.within(other, tolerance=tolerance):
            return 100.0
        return (self.overlap(other, tolerance=tolerance) / self.get_duration()) * 100

    def has_enough_overlap(self, other, cutoff=90, tolerance=0):
        return self.pct_overlap(other, tolerance=tolerance) >= cutoff


class TimedSentence(TimedElement):
    def __init__(self, start_time=0, end_time=0, text=""):
        super().__init__(start_time, end_time, text)

    def get_words(self):
        if not "words" in self.__dict__:
            self.words = self.text.split(" ")
        return self.words


class TimedWord(TimedElement):
    def __init__(self, start_time=0, end_time=0, text=""):
        super().__init__(start_time, end_time, text)


class TimedWordSentence(TimedElement):
    def __init__(self, words: List[TimedWord], fileid=None):
        assert type(words) == list
        start_time = words[0].start_time
        end_time = words[-1].end_time
        text = " ".join([w.text for w in words])
        super().__init__(start_time, end_time, text)
        if fileid is not None:
            self.fileid = fileid
        self.words = words

    def words_indexed(self, zipped=False):
        if zipped:
            return zip(self.words, range(0, len(self.words)))
        else:
            return [w for w in zip(self.words, range(0, len(self.words)))]
    
    def write_ctm(self, outfile):
        if self.fileid is None:
            fileid = "[MISSING]"
        else:
            fileid = self.fileid
        with open(outfile, "w") as of:
            for word in self.words:
                dur = float(word.duration / 1000)
                start = float(word.start_time / 1000)
                of.write(f"{fileid} 1 {start} {dur} {word.text} 1.0\n")
