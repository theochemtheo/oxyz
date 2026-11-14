---
name: oxyz
description: Use when choosing or understanding an oxyz API for extxyz reading, batching, schemas, typed records, writing, converters or locating implementation ownership.
---

# oxyz

Choose the representation before the API. Native arrays need no converter;
optional targets convert the same core data.

| Task | API and authoritative guide |
| --- | --- |
| Read/select frames | `read` / `iread`: [reading and selecting](../../../docs/guides/reading-and-selecting.md) |
| Dense arrays or bounded batches | `read_batch` / `iread_batch`: [batching and streaming](../../../docs/guides/batching-and-streaming.md) |
| Inspect, validate or normalise fields | `infer_schema`, `SchemaSpec`: [schemas and projection](../../../docs/guides/schemas-and-projection.md) |
| Named typed fields and optional observations | `Record`, `RecordBatch`: [typed records](../../../docs/guides/typed-records.md) |
| Distinct key-identifiable record types | Sibling unions: [mixed datasets](../../../docs/guides/mixed-datasets.md) |
| Save frames/batches | `write` / `Writer`: [writing and round-trip limits](../../../docs/guides/writing.md) |
| Compressed/archive/S3/HuggingFace input | [Source constraints](../../../docs/guides/compressed-and-remote.md) |
| ASE, metatomic or torch_sim output | Lazy target modules: [converters](../../../docs/guides/converters.md) |
| Inspect/check/freeze from a shell | [Command-line guide](../../../docs/guides/command-line.md) |

`read(path, int)` returns one frame; other selections return lists. Plain batches
need compatible fields. Validation checks without transforming; projection can
fill/discard fields. Record batch presence masks distinguish absence from filled
values. Do not infer presence from a zero or NaN.

Start a new workflow with [getting started](../../../docs/getting-started.md).
Use the [API reference](../../../docs/reference/api.md) for public names and
[data model](../../../docs/concepts/data-model.md) for value/ownership contracts.
For implementation work, use [develop-oxyz](../develop-oxyz/SKILL.md) and the
[owner/test map](../../../docs/development/architecture.md); for build/test/tour
mechanics, use [run-oxyz](../run-oxyz/SKILL.md). Durable rationale lives in the
[accepted decisions](../../../docs/development/decisions/index.md).
