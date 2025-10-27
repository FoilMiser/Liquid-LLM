from stage1 import runtime_setup


def test_attention_backend_fallback():
    runtime_setup.configure_attention_backend(None)
