from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import oxyz

DATA = Path(__file__).parent / "data"
FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
WORKERS = 8


@pytest.fixture(scope="module")
def many_frames(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("threads") / "many.xyz"
    path.write_text((DATA / "mace_ref_energy_forces_stress.xyz").read_text() * 200)
    return path


def _schema_parts(schema: oxyz.Schema) -> tuple[object, ...]:
    return (schema.columns, schema.metadata, schema.n_atoms.tolist(), str(schema))


def _workload(path: Path, out: Path, threads: int | None) -> dict[str, Any]:
    frames = oxyz.read(path, threads=threads)
    oxyz.write(out, frames, threads=threads)
    return {
        "frames": frames,
        "streamed": list(oxyz.iread(path)),
        "batch": oxyz.read_batch(path, threads=threads),
        "batches": list(oxyz.iread_batch(path, frames_per_batch=7, threads=threads)),
        "schema": _schema_parts(oxyz.infer_schema(path)),
        "slice": oxyz.read(path, index="::3", threads=threads),
        "written": out.read_bytes(),
    }


@pytest.mark.skipif(not FREE_THREADED, reason="needs a free-threaded build")
def test_the_gil_stays_disabled_through_import_reads_and_writes(
    tmp_path: Path,
) -> None:
    source = DATA / "mace_ref_energy_forces_stress.xyz"
    code = f"""
import sys
import oxyz
frames = oxyz.read({str(source)!r})
oxyz.read_batch({str(source)!r})
oxyz.infer_schema({str(source)!r})
oxyz.write({str(tmp_path / "out.xyz")!r}, frames)
print(sys._is_gil_enabled())
"""
    env = {key: value for key, value in os.environ.items() if key != "PYTHON_GIL"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert result.stdout.strip() == "False", result.stderr


@pytest.mark.parametrize("threads", [1, None])
def test_concurrent_reads_and_writes_match_serial(
    many_frames: Path, tmp_path: Path, threads: int | None
) -> None:
    expected = _workload(many_frames, tmp_path / "serial.xyz", threads)
    start = threading.Barrier(WORKERS)

    def run(worker: int) -> dict[str, Any]:
        start.wait()
        return _workload(many_frames, tmp_path / f"{worker}.xyz", threads)

    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(run, range(WORKERS)))
    for result in results:
        assert result.keys() == expected.keys()
        for key, value in expected.items():
            assert result[key] == value, key
