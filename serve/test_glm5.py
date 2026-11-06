"""GLM-5.3-Flash's serving pieces: the call translator (whole and a character at a time), the parser's events, the
pre-tokenizer rules.  No model needed."""
import unittest

from serve import glm5
from serve.frontend import OutputParser  # noqa: F401  (the parser's base)

CALL = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>"
QWEN = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>\n"
TOOLS = [{"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}]


def by_chars(text):
    t = glm5.Glm5Translator()
    return "".join(t.feed(c) for c in text) + t.flush()


class Translator(unittest.TestCase):
    def test_whole(self):
        t = glm5.Glm5Translator()
        self.assertEqual(t.feed(CALL) + t.flush(), QWEN)

    def test_a_character_at_a_time(self):
        self.assertEqual(by_chars(CALL), QWEN)

    def test_text_around(self):
        self.assertEqual(by_chars("before " + CALL + " after"), "before " + QWEN + " after")

    def test_text_with_a_lone_angle(self):
        self.assertEqual(by_chars("a < b and <tool"), "a < b and <tool")

    def test_no_arguments(self):
        self.assertEqual(by_chars("<tool_call>ping</tool_call>"), "<tool_call>\n<function=ping>\n</function>\n</tool_call>\n")

    def test_value_with_tags(self):
        got = by_chars("<tool_call>w<arg_key>c</arg_key><arg_value><b>x</b></arg_value></tool_call>")
        self.assertIn("<parameter=c>\n<b>x</b>\n</parameter>", got)

    def test_cut_inside_a_value(self):
        self.assertTrue(by_chars("<tool_call>w<arg_key>c</arg_key><arg_value>par").endswith("par\n</parameter>\n</function>\n</tool_call>\n"))


class Parser(unittest.TestCase):
    def test_call_after_reasoning(self):
        p = glm5.Glm5OutputParser(thinking=True, tools=TOOLS)
        evs = []
        for c in "thinking</think>" + CALL:
            evs += p.feed(c)
        evs += p.finish("stop")
        calls = [e for e in evs if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].call.name, "get_weather")
        self.assertEqual(calls[0].call.arguments, {"city": "Paris", "days": 3})
        self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "thinking")

    def test_force(self):
        self.assertEqual(glm5.to_glm_force("<tool_call>\n<function=get_weather>\n"), "<tool_call>get_weather")


if __name__ == "__main__":
    unittest.main()
