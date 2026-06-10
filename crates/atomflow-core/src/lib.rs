pub mod extxyz;
pub mod model;
pub mod schema;

pub use extxyz::{
    ExtxyzError, FrameIter, infer_schema, iter_frames, read_first_frame, read_frames,
};
pub use model::{Column, ColumnData, ColumnKind, Frame, Value};
// Schema types stay behind `schema::` — a diagnostic/contract surface, not
// core vocabulary like `Frame`. `infer_schema` is the root-level entry point.
