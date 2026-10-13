"""Frame selection by ASE-style index, shared by the conversion targets."""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING

from oxyz import _remote, _rust
from oxyz._frames import (
    Compression,
    Frame,
    IndexedFrames,
    _iter_all,
    _read_all,
    _read_all_sliced,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

    from oxyz._schema_match import Conformance
    from oxyz._schema_spec import Mode, SchemaSpec


def _is_streaming_only(path: str | Path, compression: Compression) -> bool:
    """Return whether the source cannot seek: a remote URL or compressed local file."""
    if _remote.is_remote(path):
        return True
    return _rust.is_compressed(str(path), compression)


_MAX_SLICE_PARTS = 3


def parse_index(index: int | str | slice) -> int | slice:
    """ASE's index grammar: an int, an int string, or a slice string."""
    if not isinstance(index, str):
        return index
    if ":" not in index:
        return int(index)
    parts = index.split(":")
    if len(parts) > _MAX_SLICE_PARTS:
        raise ValueError(f"invalid slice string: {index!r}")
    start, stop, step = (int(part) if part else None for part in (*parts, "", "")[:3])
    return slice(start, stop, step)


def _reject_member_on_plain(member: str | None) -> None:
    """Reject `member=` on a non-compressed file, as the core does on read paths."""
    if member is not None:
        raise ValueError(
            "member= is only valid for an archive (.zip/.tar/.tar.gz/.tar.zst) source"
        )


def nth_frame(
    path: str | Path,
    index: int,
    *,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> Frame:
    """Frame `index`; a negative index seeks after a scan rather than parsing all.

    On a compressed or remote source, or with a `schema`, the whole file is
    read instead.
    """
    if schema is not None:
        frames = _read_all(
            path,
            schema=schema,
            conformance=conformance,
            mode=mode,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        # Under project+warn some frames may have dropped.
        if not -len(frames) <= index < len(frames):
            raise IndexError(
                f"frame {index} out of range: {len(frames)} frames available"
            )
        return frames[index]

    if index < 0:
        if _is_streaming_only(path, compression):
            frames = _read_all(
                path,
                compression=compression,
                member=member,
                storage_options=storage_options,
            )
            if index + len(frames) < 0:
                raise IndexError(
                    f"frame {index} out of range: file has {len(frames)} frames"
                )
            return frames[index]
        _reject_member_on_plain(member)
        indexed = IndexedFrames(path)
        if index + len(indexed) < 0:
            raise IndexError(
                f"frame {index} out of range: file has {len(indexed)} frames"
            )
        return indexed.get(index + len(indexed))

    stream = _iter_all(
        path, compression=compression, member=member, storage_options=storage_options
    )
    frame = next(islice(stream, index, None), None)
    if frame is None:
        raise IndexError(f"frame {index} out of range")
    return frame


def _in_range(index: int, n_frames: int) -> int:
    """Return `index` if `0 <= index < n_frames`, else raise `IndexError`."""
    if index < 0 or index >= n_frames:
        raise IndexError(
            f"frame index {index} out of range: file has {n_frames} frames"
        )
    return index


def gathered_frames(  # noqa: PLR0913  the read/schema/projection options are the contract
    path: str | Path,
    indices: Sequence[int],
    threads: int | None = None,
    *,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> list[Frame]:
    """Read an explicit set of non-negative frame indices, in order.

    A seekable plain file fetches only those frames; with a `schema` or on a
    compressed or remote source, the whole file is read.
    """
    picks = [int(i) for i in indices]
    if schema is not None:
        frames = _read_all(
            path,
            threads=threads,
            schema=schema,
            conformance=conformance,
            mode=mode,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        return [frames[_in_range(i, len(frames))] for i in picks]
    if _is_streaming_only(path, compression):
        frames = _read_all(
            path,
            threads=threads,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        return [frames[_in_range(i, len(frames))] for i in picks]
    _reject_member_on_plain(member)
    indexed = IndexedFrames(path)
    n_frames = len(indexed)
    return [indexed.get(_in_range(i, n_frames)) for i in picks]


def is_forward(frames: slice) -> bool:
    """Return whether `frames` has non-negative bounds and a positive step."""
    start, stop, step = frames.start, frames.stop, frames.step
    return all(bound is None or bound >= 0 for bound in (start, stop)) and (
        step is None or step > 0
    )


def frames_for_read(  # noqa: PLR0913  the read/schema/projection options are the contract
    path: str | Path,
    frames: slice,
    threads: int | None = None,
    *,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> Iterable[Frame]:
    """Frames for an eager list read.

    An unbounded forward slice parses the whole file on `threads` cores; a
    bounded or reverse slice stops early instead. With a `schema`, the whole
    file is read and checked before slicing, so the outcome does not depend on
    the slice.
    """
    if schema is not None:
        validated = _read_all(
            path,
            threads=threads,
            schema=schema,
            conformance=conformance,
            mode=mode,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        return validated[frames]
    if is_forward(frames) and frames.stop is None:
        return _read_all_sliced(
            path,
            slice(frames.start, None, frames.step),
            threads,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
    return sliced_frames(
        path,
        frames,
        compression=compression,
        member=member,
        storage_options=storage_options,
    )


def sliced_frames(
    path: str | Path,
    frames: slice,
    *,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> Iterator[Frame]:
    """Forward slices stream; others go via the index or a full in-memory read."""
    if is_forward(frames):
        stream = _iter_all(
            path,
            schema=schema,
            conformance=conformance,
            mode=mode,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        return islice(stream, frames.start, frames.stop, frames.step)

    if schema is not None:
        projected = _read_all(
            path,
            schema=schema,
            conformance=conformance,
            mode=mode,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        return (projected[i] for i in range(*frames.indices(len(projected))))

    if _is_streaming_only(path, compression):
        in_memory = _read_all(
            path,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        return (in_memory[i] for i in range(*frames.indices(len(in_memory))))

    _reject_member_on_plain(member)
    indexed = IndexedFrames(path)
    selected = range(*frames.indices(len(indexed)))
    return (indexed.get(i) for i in selected)
