from __future__ import annotations

import gp3bayespy.fitting as fitting


def test_importable_returns_false_when_discoverable_module_import_fails(monkeypatch):
    fitting._importable.cache_clear()
    monkeypatch.setattr(fitting, "find_spec", lambda package: object())

    def fail_import(package):
        raise RuntimeError("synthetic import failure")

    monkeypatch.setattr(fitting, "import_module", fail_import)

    assert fitting._importable("synthetic_discoverable_but_broken") is False
    fitting._importable.cache_clear()


def test_importable_returns_true_when_discoverable_module_import_succeeds(monkeypatch):
    fitting._importable.cache_clear()
    monkeypatch.setattr(fitting, "find_spec", lambda package: object())
    monkeypatch.setattr(fitting, "import_module", lambda package: object())

    assert fitting._importable("synthetic_discoverable_and_importable") is True
    fitting._importable.cache_clear()
