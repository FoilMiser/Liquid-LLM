import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from Stage_1.tools.scratchpad import Scratchpad


class ScratchpadTests(unittest.TestCase):
    def test_append_and_render(self):
        pad = Scratchpad()
        pad.append("Given", "x=2")
        pad.append("Compute", "x*x")
        rendered = pad.render()
        self.assertIn("Given: x=2", rendered)
        self.assertIn("Compute: x*x", rendered)

    def test_overflow(self):
        pad = Scratchpad(max_lines=3, template=["Given:"])
        pad.append("Compute", "1")
        with self.assertRaises(ValueError):
            pad.append("Conclude", "2")


if __name__ == "__main__":
    unittest.main()
