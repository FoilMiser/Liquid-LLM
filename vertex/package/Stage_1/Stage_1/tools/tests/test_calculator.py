import math
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from Stage_1.tools.calculator import Calculator


class CalculatorTests(unittest.TestCase):
    def test_basic_ops(self):
        calc = Calculator()
        self.assertAlmostEqual(calc.evaluate("2 + 3 * 4"), 14.0)

    def test_trig_degrees(self):
        calc = Calculator(degrees=True)
        self.assertAlmostEqual(calc.evaluate("sin(30)"), 0.5, places=4)

    def test_reject_long(self):
        calc = Calculator()
        with self.assertRaises(ValueError):
            calc.evaluate("1 + " * 300)


if __name__ == "__main__":
    unittest.main()
