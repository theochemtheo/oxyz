"""Frame selection by ASE-style index, shared across conversion layers.

The index grammar (an int, an int string, or a slice string) and the
forward-streaming vs index-backed resolution live here so every conversion
layer — `oxyz.ase`, and `oxyz.metatomic` next — selects the same frames the
same way. Selection is frame-type agnostic: it yields oxyz `Frame`s, and the
caller converts to its own target.
"""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING

import oxyz._rust as _rust
from oxyz import _remote
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
    """A source that cannot seek: a remote URL, or a compressed local file."""
    if _remote.is_remote(path):
        return True
    return _rust.is_compressed(str(path), compression)


def parse_index(index: int | str | slice) -> int | slice:
    """ASE's index grammar: an int, an int string, or a slice string."""
    if not isinstance(index, str):
        return index
    if ":" not in index:
        return int(index)
    parts = index.split(":")
    if len(parts) > 3:
        raise ValueError(f"invalid slice string: {index!r}")
    start, stop, step = (int(part) if part else None for part in (*parts, "", "")[:3])
    return slice(start, stop, step)


def _reject_member_on_plain(member: str | None) -> None:
    """A non-compressed file is never an archive, so `member=` cannot apply. The
    forward read paths reject it in the core (`MemberOnNonArchive`); the seek
    path bypasses the core, so it must reject `member=` itself to match."""
    if member is not None:
        raise ValueError(
            "member= is only valid for an archive (.zip/.tar/.tar.gz) source"
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
    """Frame `index`; negatives resolve via a scan and seek, not a full parse.

    A compressed source cannot seek, so a negative index there reads the whole
    file and indexes in memory (as ASE does), losing the partial-read shortcut.
    A remote URL is likewise non-seekable and takes the same in-memory path.
    With a `schema`, any index reads the whole file and applies the schema to
    every frame before indexing — so validation/projection behaves the same way
    whatever the index, at the cost of the partial-read shortcut.
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
        # `len(frames)` is the count actually read: under project+warn some
        # frames may have been dropped, so say "available" not "file has".
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
    """A non-negative index below `n_frames`, or an IndexError. Explicit-set
    selection (`read(path, [i, j, ...])`) is non-negative only, matching
    `read_batch`; a negative index is rejected rather than wrapped."""
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
    """An explicit set of frames, in the given order (repeats allowed).

    With a `schema`, or on a streaming-only (compressed/remote) source, the
    whole file is read once and the requested frames are picked; a seekable
    plain file uses the byte-offset index to fetch only those frames. Indices
    are non-negative, matching `read_batch`.
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
    """A slice reads front-to-back iff its bounds are non-negative and its step
    positive; anything else (negative bound or step) must resolve via the index."""
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

    An unbounded forward slice needs every frame to end of file, so parse the
    whole file on all cores (`threads`) rather than streaming on one. Bounded or
    reverse slices keep the streaming/indexed path, which stops early — an eager
    read must not parse past the frames a bounded slice asks for, and `threads`
    does not apply to it.

    With a `schema`, the whole file is read and the schema applied to every
    frame, then the slice taken — so a schema error (or a projected drop) is
    handled the same way whatever the slice's shape, rather than depending on
    which frames the slice happens to traverse.
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
    """Forward slices stream; negative bounds or steps go via the index (or, on a
    compressed or remote source, or with a `schema`, via a full in-memory read)."""
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

    # A reverse/negative slice needs random access. With a schema we read+apply
    # it eagerly (the sought frames must be validated/projected); otherwise a
    # streaming-only source reads in memory and a seekable one uses the index.
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
