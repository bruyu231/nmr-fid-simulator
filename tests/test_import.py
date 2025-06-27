import importlib
def test_script_imports():
    assert importlib.import_module("nmr_simulator") is not None
