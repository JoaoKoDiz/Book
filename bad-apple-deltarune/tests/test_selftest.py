"""End-to-end validation on synthetic videos (slow, ~5-10 min):  python -m pytest -q tests -m slow"""
import json

import pytest

from badapple_dr.selftest import main


@pytest.mark.slow
def test_end_to_end_selftest(tmp_path):
    rc = main(str(tmp_path / "st"))
    rep = json.loads((tmp_path / "st" / "selftest_report.json").read_text())
    failed = {k: v for k, v in rep["checks"].items() if not v["pass"]}
    assert rc == 0, failed
