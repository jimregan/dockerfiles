# Copyright (c) 2022, 2024 Jim O'Regan for Språkbanken Tal
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
from .elements import TimedWordSentence, TimedWord
from pathlib import Path
import json


class WhisperXWord(TimedWord):
    def __init__(self, start_time=0, end_time=0, text="", speaker=None):
        super().__init__(start_time, end_time, text)
        if speaker:
            self.speaker = speaker


class WhisperXJSON(TimedWordSentence):
    def __init__(self, data=None, filename=""):
        words = []
        fileid = None
        if data is None and filename == "":
            raise ValueError("Either 'data' or 'filename' must be provided")
        elif data is not None and filename != "":
            words = self._grab(data)
            fileid = Path(filename).stem
        elif data is None:
            words = self._load(filename)
            fileid = Path(filename).stem
        else:
            words = self._grab(data)
            fileid = None
        super().__init__(words, fileid=fileid)

    def _load(self, filename):
        with open(filename) as jsonf:
            data = json.load(jsonf)
            if "segments" not in data:
                raise ValueError(f"File {filename} does not appear to contain WhisperX JSON")
            return self._grab(data)

    def _grab(self, data):
        words = []
        if type(data) == str:
            data = json.loads(data)
        if "segments" not in data or type(data["segments"]) != list:
            raise ValueError("Data does not appear to contain WhisperX JSON")
        for segment in data["segments"]:
            segment_speaker = segment.get("speaker")
            for word in segment.get("words", []):
                start = word.get("start")
                end = word.get("end")
                if start is None or end is None:
                    continue
                words.append(WhisperXWord(
                    start_time=int(start * 1000),
                    end_time=int(end * 1000),
                    text=word.get("word", ""),
                    speaker=word.get("speaker") or segment_speaker,
                ))
        return words
