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
    iter_frames,
    read_frames,
    read_frames_sliced,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path


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
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> Frame:
    """Frame `index`; negatives resolve via a scan and seek, not a full parse.

    A compressed source cannot seek, so a negative index there reads the whole
    file and indexes in memory (as ASE does), losing the partial-read shortcut.
    A remote URL is likewise non-seekable and takes the same in-memory path.
    """
    if index < 0:
        if _is_streaming_only(path, compression):
            frames = read_frames(
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

    frame = next(
        islice(
            iter_frames(
                path,
                compression=compression,
                member=member,
                storage_options=storage_options,
            ),
            index,
            None,
        ),
        None,
    )
    if frame is None:
        raise IndexError(f"frame {index} out of range")
    return frame


def is_forward(frames: slice) -> bool:
    """A slice reads front-to-back iff its bounds are non-negative and its step
    positive; anything else (negative bound or step) must resolve via the index."""
    start, stop, step = frames.start, frames.stop, frames.step
    return all(bound is None or bound >= 0 for bound in (start, stop)) and (
        step is None or step > 0
    )


def frames_for_read(
    path: str | Path,
    frames: slice,
    threads: int | None = None,
    *,
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
    """
    if is_forward(frames) and frames.stop is None:
        return read_frames_sliced(
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
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> Iterator[Frame]:
    """Forward slices stream; negative bounds or steps go via the index (or, on a
    compressed or remote source, via a full in-memory read)."""
    if is_forward(frames):
        return islice(
            iter_frames(
                path,
                compression=compression,
                member=member,
                storage_options=storage_options,
            ),
            frames.start,
            frames.stop,
            frames.step,
        )

    if _is_streaming_only(path, compression):
        in_memory = read_frames(
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
