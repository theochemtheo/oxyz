"""Declare a record, then read frames and batches as typed objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

import oxyz
from oxyz import Cell, Column, Meta, Record, RecordBatch

MIXED = Path(__file__).parent / "data" / "mixed.extxyz"

type Array = npt.NDArray[np.float64]


class Sample(Record):
    positions: Column[Array] = Column("pos", width=3)
    forces: Column[Array | None] = Column(width=3)
    energy: Meta[float] = Meta()
    cell: Cell[Array] = Cell()


class SampleBatch(RecordBatch[Sample]):
    positions: Array
    forces: Array
    energy: Array
    cell: Array


def main() -> None:
    # Frames read as records. `forces` is optional: None where a frame has none.
    samples = oxyz.read(MIXED, record=Sample)
    print("energies:", [sample.energy for sample in samples])
    print("forces present:", [sample.forces is not None for sample in samples])

    # A generic batch: fields are looked up by key, and optional fields are
    # NaN-filled where absent, with a mask saying which frames had them.
    batch = oxyz.read_batch(MIXED, record=RecordBatch[Sample])
    print("energy:", batch[Sample.energy])
    print("forces shape:", batch[Sample.forces].shape)
    print("forces present:", batch.present(Sample.forces))
    print("third frame:", batch[2])

    # A declared batch has the fields as typed attributes as well.
    declared = oxyz.read_batch(MIXED, record=SampleBatch)
    print("cell shape:", declared.cell.shape)

    # A record from a frame already in memory, checked against the fields.
    frame = oxyz.read(MIXED, 0)
    print("from_frame matches read:", Sample.from_frame(frame) == samples[0])


if __name__ == "__main__":
    main()
