# Accepted decisions

These records describe supported current contracts and evidence in tracked
source/tests. They are not reconstructions of private discussions or claims
about when a choice was first made. Recommendations for future changes must be
labelled as proposals, not silently folded into accepted rationale.

| Decision | Status | Contract |
| --- | --- | --- |
| [Lossless core and layering](lossless-core-and-layering.md) | Accepted | Preserve file data; keep converter conventions outside Rust |
| [Schemas and projection](schemas-and-projection.md) | Accepted | Separate observed data from declared validation/projection policy |
| [Typed records and unions](typed-records-and-unions.md) | Accepted | Bind typed fields to the shared pipeline; preserve presence and member identity |
| [Optional output targets](optional-output-targets.md) | Accepted | Isolate converters and remote dependencies behind extras/lazy imports |

## Changing a decision

A substantive replacement gets a new descriptive filename. Mark the old record
`Superseded`, link forward to its replacement and add a reciprocal
`Supersedes` link on the new record. Use `Deprecated` for a contract being
retired without a replacement. Keep the old rationale readable rather than
deleting it. No invented adoption dates or private planning links are needed.

Return to the [architecture map](../architecture.md) for implementation owners
and [development](../index.md) for the change workflow.
