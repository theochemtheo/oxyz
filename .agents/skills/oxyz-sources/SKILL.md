---
name: oxyz-sources
description: Use when changing oxyz selectors, compressed files, archive members, remote transport, S3 or HuggingFace authentication, source ordering or seek restrictions.
---

# oxyz sources

Choose the selection path from the source's capabilities, not its apparent
filename. Start with the [root router](../oxyz/SKILL.md),
[source guide](../../../docs/guides/compressed-and-remote.md),
[selection contract](../../../docs/guides/reading-and-selecting.md) and
[storage concepts](../../../docs/concepts/storage-and-streaming.md).

## Owners and checks

Python paths below are under `src/oxyz/`; Python tests are under `tests/`.

| Concern | Owner | Existing checks |
| --- | --- | --- |
| Integer/slice/gather selection and streaming-only dispatch | `_select.py`, `_frames.py`, `_batch.py` | `test_extxyz.py`, `test_batch.py`, `test_remote_unit.py` |
| URL rewriting, authentication, codec sniffing and obstore adapters | `_remote.py` | `test_remote_unit.py`; local-mock integration in `test_remote.py` |
| Codec/member decoding and joined stream boundaries | `crates/oxyz-core/src/decode.rs` | Core `compressed.rs`, `decode_sources.rs`; Python `test_compressed.py` |
| Structural frame offsets and indexed reads | `crates/oxyz-core/src/index.rs` | Core `index_scan.rs`, `indexed_frames.rs` |
| Chunked-reader binding and archive source factories | `crates/oxyz-py/src/lib.rs` | `test_reader_entries.py`, `test_remote.py` |

Core test paths are under `crates/oxyz-core/tests/`. Use
[oxyz-core](../oxyz-core/SKILL.md) for parser/index invariants.

## Invariants and traps

- Local uncompressed files can seek by structural index. Compressed local and
  all remote inputs use streaming-only frame selection. A scan does not grant
  random access; negative/reverse selections may materialise the whole file.
- `iread_batch(..., frames_per_batch=...)` can stream. `shuffle`,
  `atoms_per_batch` and `memory_scales_with` require random access and are
  rejected on streaming-only sources. Use a local **uncompressed** copy;
  never hide a full-file buffer to bypass that restriction.
- Codec inference prefers recognised suffixes, then magic bytes. Tar layers
  need an explicit override when the name no longer identifies them.
  ZIP needs seek/range support internally; remote tar uses a callable stream
  factory for separate enumeration/read passes and can fetch more than once.
- `member` is archive-only: absent means exactly one eligible entry, an exact
  name selects one entry, `"*"` joins eligible entries in stored order, not
  filename order. Members cannot complete another member's truncated frame.
  Joined parser locations are logical-stream positions, not per-member offsets.
- Remote reads need the `s3` extra (`obstore`). Keep client/provider options
  separate. S3 uses provider configuration/`AWS_*`; Hub bearer precedence is
  explicit case-insensitive authorisation header, `HF_TOKEN`, then
  `HUGGING_FACE_HUB_TOKEN`. Never forward Hub tokens to other stores.
  Hub repo files and S3-compatible Hub Storage Buckets are distinct products.
- Use unit fakes and the loopback moto `s3_store` fixture with dummy credentials.
  Do not require live credentials, enable `OXYZ_TEST_HF`, or claim GCS/Azure
  integration coverage from S3 mocks. Restore patched environment/import state.

## Targeted verification

For an S3 compressed archive, use native `read`/`iread` with `member=` and
`storage_options=`; choose fixed-frame `iread_batch` for streaming batches.
Verify archive/transport composition with the existing local mock:

```sh
uv run pytest -q tests/test_remote.py -k test_read_frames_remote_all_members
```

For selector order/duplicates and negative indices inspect the corresponding
`test_remote_frame_gathers_*` and `test_remote_streamed_negative_*` cases in
`test_remote_unit.py`. That module also checks auth precedence, codec sniffing
and `test_iread_batch_remote_rejects_random_access`. `test_compressed.py`
checks all-members ordering and compressed batching restrictions. Run core
codec tests with `cargo test -p oxyz-core --test decode_sources --test compressed`.
Use [run-oxyz](../run-oxyz/SKILL.md) for environment mechanics; run only affected
checks, not every routed suite for a prose edit.
