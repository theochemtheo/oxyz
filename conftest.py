"""Collect public documentation examples without changing ordinary tests."""

from __future__ import annotations

import os
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_ROOT = Path(__file__).resolve().parent
_FLAGS = NORMALIZE_WHITESPACE | ELLIPSIS


@pytest.fixture(scope="module")
def _doc_workdir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    previous = Path.cwd()
    workdir = tmp_path_factory.mktemp("doc-example")
    try:
        os.chdir(workdir)
        yield workdir
    finally:
        os.chdir(previous)


def _source_setup(namespace: dict[str, object]) -> None:
    namespace["DATA"] = _ROOT / "examples" / "data"


@cache
def _document_collectors() -> tuple[
    Callable[[Path, pytest.Collector], pytest.File | None],
    Callable[[Path, pytest.Collector], pytest.File | None],
]:
    from sybil import Sybil
    from sybil.document import PythonDocStringDocument
    from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser
    from sybil.parsers.rest import DocTestParser

    markdown = Sybil(
        parsers=[PythonCodeBlockParser(doctest_optionflags=_FLAGS), SkipParser()],
        pattern="*.md",
        fixtures=["_doc_workdir"],
    )
    source = Sybil(
        parsers=[DocTestParser(optionflags=_FLAGS)],
        pattern="*.py",
        document_types={".py": PythonDocStringDocument},
        setup=_source_setup,
        fixtures=["_doc_workdir"],
    )
    return markdown.pytest(), source.pytest()


def pytest_collect_file(
    file_path: Path, parent: pytest.Collector
) -> pytest.File | None:
    """Route only public Markdown and package docstrings to Sybil."""
    relative = file_path.relative_to(_ROOT)
    if relative == Path("README.md") and not (
        parent.session.isinitpath(file_path) or parent.session.isinitpath(_ROOT)
    ):
        return None
    if relative.suffix == ".py" and relative.parts[:2] == ("src", "oxyz"):
        return _document_collectors()[1](file_path, parent)
    if relative == Path("README.md") or (
        relative.suffix == ".md"
        and relative.parts[0] == "docs"
        and relative.parts[:2] != ("docs", "superpowers")
    ):
        return _document_collectors()[0](file_path, parent)
    return None
