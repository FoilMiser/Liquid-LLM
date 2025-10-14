from stage1.tool_use.calculator import SafeCalculator
from stage1.tool_use.scratchpad import Scratchpad
from stage1.tool_use.traces import maybe_inject_tool_result


def test_tool_pipeline():
    calc = SafeCalculator()
    result = calc.evaluate("sin(pi/2)")
    assert abs(float(result) - 1.0) < 1e-6

    scratch = Scratchpad(max_lines=2)
    scratch.add("line1")
    scratch.add("line2")
    assert scratch.render() == "line1\nline2"
    scratch.add("line3")
    assert scratch.render() == "line2\nline3"

    text = "CALL calculator:\"2+2\""
    injected = maybe_inject_tool_result(text)
    assert "RESULT:" in injected
