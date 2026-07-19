"""tests/test_promotion_check_skip_is_not_pass.py — PR#25 (AUDIT2-2).

External audit BD601 (round 2) flagged that scripts/promotion_check.py counted a SKIPPED required
node as a PASS (pytest exits 0 on skip, and the gate runner only checked returncode). The fix parses
JUnit XML and requires every selected testcase to be a genuine pass. These tests lock that: a skip
(synthetic XML and a real pytest run) must NOT count as a pass, so a gate cannot go green on a skip.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "promotion_check.py"


def _load_promotion_check():
    spec = importlib.util.spec_from_file_location("promotion_check_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: the module defines a @dataclass whose field-type resolution looks the
    # module up in sys.modules (cls.__module__); without this, collection errors with AttributeError.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_PC = _load_promotion_check()
_junit_all_passed = _PC._junit_all_passed


_PASS_XML = """<?xml version="1.0"?>
<testsuites><testsuite name="pytest" tests="2">
  <testcase classname="t" name="a" time="0.1"/>
  <testcase classname="t" name="b" time="0.1"/>
</testsuite></testsuites>"""

_SKIP_XML = """<?xml version="1.0"?>
<testsuites><testsuite name="pytest" tests="2">
  <testcase classname="t" name="a" time="0.1"/>
  <testcase classname="t" name="b"><skipped message="cond"/></testcase>
</testsuite></testsuites>"""

_FAIL_XML = """<?xml version="1.0"?>
<testsuites><testsuite name="pytest" tests="1">
  <testcase classname="t" name="a"><failure message="boom"/></testcase>
</testsuite></testsuites>"""

_ERROR_XML = """<?xml version="1.0"?>
<testsuites><testsuite name="pytest" tests="1">
  <testcase classname="t" name="a"><error message="setup"/></testcase>
</testsuite></testsuites>"""

_EMPTY_XML = """<?xml version="1.0"?>
<testsuites><testsuite name="pytest" tests="0"></testsuite></testsuites>"""


def _write(tmp_path, xml):
    p = tmp_path / "j.xml"
    p.write_text(xml, encoding="utf-8")
    return p


def test_all_pass_is_pass(tmp_path):
    assert _junit_all_passed(_write(tmp_path, _PASS_XML)) is True


@pytest.mark.parametrize("xml", [_SKIP_XML, _FAIL_XML, _ERROR_XML, _EMPTY_XML])
def test_skip_fail_error_empty_are_not_pass(tmp_path, xml):
    assert _junit_all_passed(_write(tmp_path, xml)) is False


def test_missing_or_malformed_xml_is_not_pass(tmp_path):
    assert _junit_all_passed(tmp_path / "does_not_exist.xml") is False
    bad = tmp_path / "bad.xml"
    bad.write_text("not xml <<<", encoding="utf-8")
    assert _junit_all_passed(bad) is False


def test_real_pytest_skip_is_not_counted_as_pass(tmp_path):
    """End-to-end: a real pytest SKIP produces a <skipped> testcase that _junit_all_passed rejects,
    while a real pass is accepted -- so a skipped required node can never green a gate."""
    junit = tmp_path / "real.xml"
    skip_t = tmp_path / "t_skip.py"
    skip_t.write_text("import pytest\n\ndef test_s():\n    pytest.skip('unavailable')\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "pytest", str(skip_t), f"--junit-xml={junit}", "-q",
                    "-p", "no:cacheprovider"], capture_output=True, text=True)
    assert junit.exists()
    assert _junit_all_passed(junit) is False, "a real pytest skip must NOT count as a pass"

    junit2 = tmp_path / "real_pass.xml"
    pass_t = tmp_path / "t_pass.py"
    pass_t.write_text("def test_p():\n    assert True\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "pytest", str(pass_t), f"--junit-xml={junit2}", "-q",
                    "-p", "no:cacheprovider"], capture_output=True, text=True)
    assert _junit_all_passed(junit2) is True
