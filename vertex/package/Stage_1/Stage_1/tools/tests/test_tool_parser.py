import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from Stage_1.tools.tool_trace_injector import ToolTraceInjector


class ToolTraceInjectorTests(unittest.TestCase):
    def test_injects_calculator(self):
        injector = ToolTraceInjector()
        tokens = ["Question", 'CALL calculator:"2+2"']
        out = injector.inject(tokens)
        self.assertTrue(any(t.startswith("RESULT:") for t in out))

    def test_limits_calls(self):
        injector = ToolTraceInjector(max_calls=1)
        tokens = ['CALL calculator:"1+1"', 'CALL calculator:"2+2"']
        out = injector.inject(tokens)
        self.assertEqual(sum(t.startswith("RESULT:") for t in out), 1)


if __name__ == "__main__":
    unittest.main()
