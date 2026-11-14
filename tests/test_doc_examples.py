from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

pytest_plugins = ["pytester"]

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(pytester, monkeypatch):
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pytester.makeconftest((_ROOT / "conftest.py").read_text())
    pytester.makeini("[pytest]\naddopts = -p no:doctest\n")
    return pytester


def _page(project, name, content):
    path = project.path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip())
    return path


def test_wrong_markdown_output_reports_source(project):
    _page(project, "docs/wrong.md", "```python\n>>> 2 + 2\n5\n```\n")
    result = project.runpytest_subprocess("docs/wrong.md", "-q")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*wrong.md:2*"])
    result.stdout.fnmatch_lines(["*Expected:*", "*5*", "*Got:*", "*4*"])


@pytest.mark.parametrize("source", [">>> 2 + 2\n5", "assert 2 + 2 == 5"])
@pytest.mark.parametrize(
    "fence",
    [
        'python title="Addition" linenums="1" hl_lines="1"',
        'python {.no-copy title="Addition"}',
    ],
)
@pytest.mark.parametrize(
    ("container", "prefix", "expected_line"),
    [
        ("", "", 2),
        ('!!! example "Checked example"', "    ", 4),
        ('=== "Python"', "    ", 4),
    ],
)
def test_decorated_python_fences_report_source(
    project, source, fence, container, prefix, expected_line
):
    header = f"{container}\n\n" if container else ""
    block = "\n".join(
        prefix + row for row in [f"```{fence}", *source.splitlines(), "```"]
    )
    _page(project, "docs/decorated.md", header + block + "\n")
    result = project.runpytest_subprocess("docs/decorated.md", "-q")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines([f"*decorated.md:{expected_line}*"])
    if source.startswith(">>>"):
        result.stdout.fnmatch_lines(["*Expected:*", "*5*", "*Got:*", "*4*"])


def test_markdown_blocks_share_only_document_state(project):
    _page(
        project,
        "docs/first.md",
        """
        ```python
        value = 7
        ```

        ```python
        >>> value
        7
        ```
        """,
    )
    _page(
        project,
        "docs/second.md",
        """
        ```python
        assert "value" not in globals()
        assert "DATA" not in globals()
        ```
        """,
    )
    result = project.runpytest_subprocess("docs/first.md", "docs/second.md", "-q")
    result.assert_outcomes(passed=3)


def test_document_workdir_is_isolated_and_restored(project):
    _page(
        project,
        "docs/first.md",
        f"""
        ```python
        from pathlib import Path
        assert Path.cwd() != Path({str(project.path)!r})
        Path("out.xyz").write_text("document output")
        Path({str(project.path / "document-cwd")!r}).write_text(str(Path.cwd()))
        ```
        """,
    )
    _page(
        project,
        "docs/second.md",
        """
        ```python
        from pathlib import Path
        assert not Path("out.xyz").exists()
        ```
        """,
    )
    _page(
        project,
        "docs/failing.md",
        """
        ```python
        from pathlib import Path
        Path("failure.xyz").write_text("failed document output")
        raise ValueError("deliberate document failure")
        ```
        """,
    )
    project.makepyfile(
        test_after=f"""
        from pathlib import Path

        def test_normal_cwd():
            assert Path.cwd() == Path({str(project.path)!r})
            assert not Path("out.xyz").exists()
            assert not Path("failure.xyz").exists()
        """
    )
    _page(
        project,
        "finish_plugin.py",
        f"""
        from pathlib import Path

        def pytest_sessionfinish():
            Path({str(project.path / "final-cwd")!r}).write_text(str(Path.cwd()))
        """,
    )
    result = project.runpytest_subprocess(
        "docs/first.md",
        "docs/second.md",
        "docs/failing.md",
        "test_after.py",
        "-p",
        "finish_plugin",
        "-q",
    )
    result.assert_outcomes(passed=3, failed=1)
    assert (project.path / "final-cwd").read_text() == str(project.path)
    workdir = Path((project.path / "document-cwd").read_text())
    assert (workdir / "out.xyz").read_text() == "document output"
    assert not (project.path / "out.xyz").exists()
    assert not (project.path / "failure.xyz").exists()


@pytest.mark.parametrize(
    "fence",
    ["python", 'python title="Schematic"', 'python {.no-copy title="Schematic"}'],
)
def test_reasoned_skip_is_reported(project, fence):
    _page(
        project,
        "docs/schematic.md",
        f"""
        <!-- skip: next "requires private remote credentials" -->

        ```{fence}
        >>> undefined_remote_client.fetch()
        unavailable output
        ```
        """,
    )
    result = project.runpytest_subprocess("docs/schematic.md", "-q", "-rs")
    assert result.ret == 0
    assert result.parseoutcomes()["skipped"] == 1
    result.stdout.fnmatch_lines(["*requires private remote credentials*"])


def test_source_docstrings_are_checked_once(project):
    _page(project, "examples/data/calls", "")
    _page(project, "src/oxyz/__init__.py", "")
    module = _page(
        project,
        "src/oxyz/sample.py",
        '''
        r"""A source example using module names and legacy data.

        >>> with (DATA / "calls").open("a") as output:
        ...     _ = output.write("x")
        >>> answer
        42
        >>> print("alpha   beta\\nrest of output")
        alpha beta
        rest...
        """

        answer = 42
        incidental = ">>> 1 / 0"
        ''',
    )
    _page(project, ".agents/skills/example/SKILL.md", "```python\n1 / 0\n```\n")
    _page(project, "docs/superpowers/plan.md", "```python\n1 / 0\n```\n")
    _page(project, "unrelated.md", "```python\n1 / 0\n```\n")
    project.makepyfile(
        test_ordinary='def test_ok():\n    text = ">>> 1 / 0"\n    assert text\n'
    )
    result = project.runpytest_subprocess(".", "-q")
    assert result.ret == 0
    assert (project.path / "examples/data/calls").read_text() == "x"
    module.write_text(module.read_text().replace("\n42\n", "\n43\n"))
    result = project.runpytest_subprocess("src/oxyz/sample.py", "-q")
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(
        ["*Expected:*", "*43*", "*Got:*", "*42*", "*sample.py:*"]
    )


def test_non_document_collection_needs_no_sybil(project):
    _page(project, "README.md", "```python\n1 / 0\n```\n")
    _page(
        project,
        "block_sybil.py",
        """
        import sys
        from importlib.abc import MetaPathFinder

        class BlockSybil(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "sybil" or fullname.startswith("sybil."):
                    raise ImportError("Sybil is absent in this environment")

        sys.meta_path.insert(0, BlockSybil())
        """,
    )
    _page(
        project,
        "tests/test_ordinary.py",
        """
        import sys

        def test_no_optional_imports():
            assert "sybil" not in sys.modules
            assert "oxyz.ase" not in sys.modules
            assert "oxyz.metatomic" not in sys.modules
            assert "oxyz.torch_sim" not in sys.modules
        """,
    )
    result = project.runpytest_subprocess("tests", "-p", "block_sybil", "-q")
    result.assert_outcomes(passed=1)
