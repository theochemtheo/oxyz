pub mod batch;
pub mod decode;
pub mod extxyz;
pub mod index;
pub mod model;
pub mod schema;

pub use batch::{Batch, BatchBuilder, BatchError};
pub use decode::{Compression, DecodedReader, is_compressed, open_decoded};
pub use extxyz::{
    BatchIter, ExtxyzError, FrameIter, IndexedFrames, infer_schema, infer_schema_from,
    iter_batches, iter_batches_from, iter_frames, iter_frames_from, read_all_batch,
    read_all_batch_from, read_batch, read_batch_from, read_first_frame, read_frames, scan_frames,
    scan_frames_with_volume, scan_index, scan_index_with_volume,
};
#[cfg(feature = "parallel")]
pub use extxyz::{
    read_all_batch_parallel, read_all_batch_parallel_from, read_batch_parallel,
    read_batch_parallel_from, read_frames_parallel, read_frames_parallel_from,
};
pub use model::{Column, ColumnData, ColumnKind, Frame, Value};
// Schema and index types stay behind `schema::` / `index::` — derived
// surfaces, not core vocabulary like `Frame`. The entry points (readers and
// `infer_schema` / `scan_index`) live at the root.
