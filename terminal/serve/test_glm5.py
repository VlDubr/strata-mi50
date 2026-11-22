"""GLM-5.3-Flash's serving pieces: the call translator (whole and a character at a time), the parser's events, the
pre-tokenizer rules, the thinking switch and the size of a picture.  No model needed."""
import unittest

from serve import frontend, glm5
from serve.ds_pictures import Glm5Pictures, glm5_canvas
from serve.frontend import OutputParser  # noqa: F401  (the parser's base)
from serve.server import Service

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


class Thinking(unittest.TestCase):
    def setUp(self):
        frontend.use_family("glm5-next")
        self.addCleanup(frontend.use_family, None)

    def test_none_is_off(self):
        for word in ("none", "off", "disabled", False):
            self.assertEqual(frontend.effort_kwargs(word), {"enable_thinking": False})

    def test_levels_stay(self):
        self.assertEqual(frontend.effort_kwargs("low"), {"reasoning_effort": "low"})
        self.assertEqual(frontend.effort_kwargs("medium"), {"reasoning_effort": "high"})

    def render(self, kwargs):
        class Template:
            seen = {}

            def render(self, messages, tools=None, **kw):
                Template.seen = kw
                return "[gMASK]<sop><|user|>hi<|assistant|><think>"

        s = Service.__new__(Service)
        s.family, s.template, s.effort_end = "glm5-next", Template(), False
        return s.render_prompt([], None, kwargs), Template.seen

    def test_off_closes_the_block(self):
        text, seen = self.render({"enable_thinking": False})
        self.assertTrue(text.endswith("<think></think>"))
        self.assertEqual(seen, {"reasoning_effort": "low"})

    def test_on_leaves_it_open(self):
        text, _ = self.render({"reasoning_effort": "high"})
        self.assertTrue(text.endswith("<think>"))


class Picture(unittest.TestCase):
    MIN, MAX, CELL = 12544, 4096 * 28 * 28, 28

    def canvas(self, w, h):
        return glm5_canvas(w, h, self.CELL, self.MIN, self.MAX)

    def test_aligned_up_and_not_scaled(self):
        self.assertEqual(self.canvas(300, 200), (308, 224, 300, 200))

    def test_small_is_scaled_up_to_the_budget(self):
        cw, ch, w, h = self.canvas(50, 50)
        self.assertGreaterEqual(cw * ch, self.MIN)
        self.assertGreater(w, 50)

    def test_large_fits_the_budget(self):
        cw, ch, w, h = self.canvas(8000, 6000)
        self.assertLessEqual(cw * ch, self.MAX)
        self.assertEqual((cw % 28, ch % 28), (0, 0))
        self.assertLessEqual(w, cw)
        self.assertLessEqual(h, ch)

    def test_span_is_a_token_a_block(self):
        p = Glm5Pictures(154854, 4096, self.MIN, None, 14, 2)
        self.assertEqual(p.span(308, 224), 11 * 8)


if __name__ == "__main__":
    unittest.main()
