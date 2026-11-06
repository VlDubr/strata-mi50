"""GLM-5.3-Flash's tool calls: `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`.

The server reads calls in the Qwen form (`<tool_call>`, `<function=NAME>`, `<parameter=K>` ...), so this file only translates
the model's text into that form while it streams; everything else OutputParser does (reasoning, declared tools, recovery,
argument coercion by the schema) stays as it is.  The same pattern as serve/dsml.py.
"""
from __future__ import annotations

from serve.frontend import OutputParser

CALL_OPEN, CALL_CLOSE = "<tool_call>", "</tool_call>"
KEY_OPEN, KEY_CLOSE = "<arg_key>", "</arg_key>"
VAL_OPEN, VAL_CLOSE = "<arg_value>", "</arg_value>"


def _held(text: str, tag: str) -> int:
    """How many characters at the end of `text` may be the start of `tag`."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


class Glm5Translator:
    """The model's call block -> the Qwen form, a delta at a time.  What may be the start of a tag is held back."""

    def __init__(self):
        self.buf = ""
        self.state = "text"      # text, name, between, key, value_gap, value

    @property
    def held(self) -> bool:
        return bool(self.buf) or self.state != "text"

    def feed(self, delta: str) -> str:
        self.buf += delta
        out: list[str] = []
        while self._step(out):
            pass
        return "".join(out)

    def flush(self) -> str:
        """The turn ended: what is held goes out as it stands, and an open call is closed so the parser can judge it."""
        out = [self.buf]
        self.buf = ""
        if self.state == "value":
            out.append("\n</parameter>\n</function>\n</tool_call>\n")
        elif self.state != "text":
            out.append("</function>\n</tool_call>\n")
        self.state = "text"
        return "".join(out)

    def _step(self, out: list[str]) -> bool:
        s = self.buf
        if self.state == "text":
            at = s.find(CALL_OPEN)
            if at >= 0:
                out.append(s[:at])
                self.buf = s[at + len(CALL_OPEN):]
                self.state = "name"
                return True
            keep = _held(s, CALL_OPEN)
            out.append(s[:len(s) - keep])
            self.buf = s[len(s) - keep:]
            return False
        if self.state == "name":
            ends = [p for p in (s.find(KEY_OPEN), s.find(CALL_CLOSE)) if p >= 0]
            if not ends:
                return False
            at = min(ends)
            out.append("<tool_call>\n<function=%s>\n" % s[:at].strip())
            self.buf = s[at:]
            self.state = "between"
            return True
        if self.state == "between":
            t = s.lstrip()
            if t.startswith(KEY_OPEN):
                self.buf = t[len(KEY_OPEN):]
                self.state = "key"
                return True
            if t.startswith(CALL_CLOSE):
                self.buf = t[len(CALL_CLOSE):]
                self.state = "text"
                out.append("</function>\n</tool_call>\n")
                return True
            if KEY_OPEN.startswith(t) or CALL_CLOSE.startswith(t):
                self.buf = t
                return False
            self.buf = t         # (not the form: its text goes on as the next call's or the answer's)
            self.state = "text"
            out.append("</function>\n</tool_call>\n")
            return True
        if self.state == "key":
            at = s.find(KEY_CLOSE)
            if at < 0:
                return False
            out.append("<parameter=%s>\n" % s[:at].strip())
            self.buf = s[at + len(KEY_CLOSE):]
            self.state = "value_gap"
            return True
        if self.state == "value_gap":
            t = s.lstrip()
            if t.startswith(VAL_OPEN):
                self.buf = t[len(VAL_OPEN):]
                self.state = "value"
                return True
            self.buf = t
            return False
        at = s.find(VAL_CLOSE)     # value
        if at >= 0:
            out.append(s[:at] + "\n</parameter>\n")
            self.buf = s[at + len(VAL_CLOSE):]
            self.state = "between"
            return True
        keep = _held(s, VAL_CLOSE)
        out.append(s[:len(s) - keep])
        self.buf = s[len(s) - keep:]
        return False


class Glm5OutputParser(OutputParser):
    """OutputParser for GLM-5.3: the model's calls are read as the template-form calls they are."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.glm = Glm5Translator()

    def feed(self, delta: str):
        return super().feed(self.glm.feed(delta))

    def finish(self, reason: str | None = None):
        rest = self.glm.flush()
        out = super().feed(rest) if rest else []
        return out + super().finish(reason)

    @property
    def clean(self) -> bool:
        return not self.buf and not self.glm.held


def to_glm_force(force: str | None) -> str | None:
    """forced_call's opening of a call in the Qwen form -> the same opening in the model's spelling."""
    if not force:
        return force
    marker = "<function="
    at = force.find(marker)
    name = force[at + len(marker):] if at >= 0 else ""
    return CALL_OPEN + (name[:-2] if name.endswith(">\n") else name)
