from stage1.tool_use import calculator, registry, scratchpad, traces


def test_calculator_safe_expressions():
    assert calculator.evaluate("sin(pi/2)") == "1"
    assert calculator.evaluate("__import__('os')") == "ERROR:FORBIDDEN"
    assert calculator.evaluate("open('foo')") == "ERROR:FORBIDDEN"


def test_scratchpad_cap():
    pad = scratchpad.new_scratchpad(max_lines=2)
    pad.add("step1", "value1")
    pad.add("step2", "value2")
    pad.add("step3", "value3")
    rendered = pad.render()
    assert "step1" not in rendered
    assert "step2" in rendered and "step3" in rendered


def test_trace_injection():
    text = "CALL calculator: \"1+1\""
    injected = traces.maybe_inject_tool_result(text)
    assert "RESULT:" in injected


def test_registry_unknown_tool():
    assert registry.call_tool("unknown", "1+1") == "ERROR:UNKNOWN_TOOL"
