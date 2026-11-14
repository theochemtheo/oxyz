# Compressed and remote sources

The frame readers, `scan`, `infer_schema` and batch readers accept compressed
paths. The same source options reach the [converter readers](converters.md),
including `SystemSource` and `SimStateSource`. Choose frames with
[reading and selection](reading-and-selecting.md); choose a bounded-memory
workflow with [batching and streaming](batching-and-streaming.md).

## Select a codec

`compression="infer"` is the default. A recognised filename suffix takes
precedence; otherwise oxyz checks leading magic bytes. The override
`compression="none"` reads bytes unchanged, even with a compressed-looking name.
Other overrides force the named decoder, independent of the filename.

| Input | Inferred suffix | Explicit `compression` | Writable |
| --- | --- | --- | --- |
| Plain extxyz | No recognised compression suffix or magic | `"none"` | Yes |
| gzip | `.gz` | `"gzip"` | Yes |
| zstd | `.zst` | `"zstd"` | No |
| ZIP | `.zip` | `"zip"` | Yes |
| tar | `.tar` | `"tar"` | Yes |
| gzip-compressed tar | `.tar.gz`, `.tgz` | `"tar.gz"` | Yes |
| zstd-compressed tar | `.tar.zst`, `.tzst` | `"tar.zst"` | No |

Suffix matching is case-insensitive. Magic detection recognises gzip, zstd and
ZIP, not tar or the tar layer inside gzip/zstd. A renamed tar archive therefore
needs an explicit tar override: `compression="tar.gz"`, not `"gzip"`, for a
renamed gzip-compressed tar. Concatenated gzip members and concatenated zstd
frames are decoded through to the end.

Decompression streams without creating a temporary decompressed file. Parsing
can still use multiple threads. This does not make an eager `read` bounded in
memory: its returned frames remain materialised. For output levels, append
support and archive buffering, see [writing trajectories](writing.md#select-a-codec).
Remote writing is not provided by these reader options.

```python title="Create and read a tiny gzip stream"
import gzip
from pathlib import Path

import oxyz

first = b"1\nProperties=species:S:1:pos:R:3 energy=-1.5\nH 0.25 0 0\n"
second = b"1\nProperties=species:S:1:pos:R:3 energy=-2.5\nHe 0 0.5 0\n"
Path("tiny.extxyz.gz").write_bytes(gzip.compress(first) + gzip.compress(second))
frames = oxyz.read("tiny.extxyz.gz")
assert [frame.metadata["energy"] for frame in frames] == [-1.5, -2.5]
assert frames[0].columns["species"] == ["H"]
assert frames[0].columns["pos"].shape == (1, 3)
assert frames[0].columns["pos"].tolist() == [[0.25, 0.0, 0.0]]

Path("compressed.bin").write_bytes(gzip.compress(first))
assert oxyz.read("compressed.bin", compression="gzip") == frames[:1]
assert oxyz.read("compressed.bin") == frames[:1]  # Magic-byte fallback.
Path("plain.gz").write_bytes(first)
assert oxyz.read("plain.gz", compression="none") == frames[:1]
```

## Select archive members

ZIP, tar, gzip-compressed tar and zstd-compressed tar accept `member=`:

- Without `member`, exactly one eligible `.xyz` or `.extxyz` entry is required.
  No match or multiple matches raises `ValueError`; ambiguous archives preview
  eligible names rather than silently choosing the first.
- `member="path/run.xyz"` selects that exact archive path, not a glob. An explicit
  name can select an entry without an extxyz suffix. A missing name raises
  `ValueError` with available names.
- `member="*"` reads all eligible entries in stored archive order, **not** sorted
  filename order. Suffix matching is case-insensitive. Directories, names under
  `__MACOSX/` and basenames starting with `._` are excluded from automatic
  selection. Other non-extxyz files are ignored.
- `member` is invalid for plain files and single gzip/zstd streams.

```python title="Create ZIP and tar archives and verify stored order"
import io
import tarfile
import zipfile

entries = [("z.extxyz", first + b"\n"), ("a.xyz", second)]
with zipfile.ZipFile("runs.zip", "w") as archive:
    archive.writestr("README.txt", "not trajectory data")
    for name, body in entries:
        archive.writestr(name, body)

with tarfile.open("runs.tar.gz", "w:gz") as archive:
    for name, body in entries:
        info = tarfile.TarInfo(name)
        info.size = len(body)
        archive.addfile(info, io.BytesIO(body))

for path in ["runs.zip", "runs.tar.gz"]:
    selected = oxyz.read(path, member="a.xyz")
    assert selected == frames[1:]
    joined = oxyz.read(path, member="*")
    assert joined == frames
    assert [frame.metadata["energy"] for frame in joined] == [-1.5, -2.5]
    assert [frame.metadata["energy"] for frame in oxyz.iread(path, member="*")] == [-1.5, -2.5]
    assert oxyz.scan(path, member="*").n_frames == 2
    try:
        oxyz.read(path)
    except ValueError as error:
        assert "multiple extxyz members" in str(error)
        assert "z.extxyz" in str(error) and "a.xyz" in str(error)
    else:
        raise AssertionError("ambiguous archive was accepted")
```

### Source positions and member boundaries

With `member="*"`, parser frame/line positions refer to the joined logical
stream, not to a separate per-member source location. A blank count line ends
its member, not the entire archive: the second member in the example above is
still read. Members cannot complete one another's partial frames. A member
ending inside a frame produces an `OSError` naming that member:

```python title="Check the source context of a truncated member"
with zipfile.ZipFile("broken.zip", "w") as archive:
    archive.writestr("good.xyz", first)
    archive.writestr("short.xyz", b"2\nProperties=species:S:1:pos:R:3\nH 0 0 0\n")

try:
    oxyz.read("broken.zip", member="*")
except OSError as error:
    assert "short.xyz" in str(error)
    assert "ends inside a frame" in str(error)
else:
    raise AssertionError("truncated archive member was accepted")
```

See [errors and source context](../reference/errors.md) for structured parser
errors; do not assume every parse error identifies an archive entry.

## Seeking and memory limits

Local uncompressed files support structural indexing and seeking. Compressed
local inputs and **all remote inputs**, even uncompressed ones, take the
streaming-only selection path:

- Forward selections can stream; negative or reverse selections may require a
  complete read into memory rather than a seek. This also applies to ASE's
  default last-frame selection. Streaming-only does not mean every selection
  has bounded memory.
- `iread_batch` with `frames_per_batch` can stream. Random-access strategies
  (`shuffle`, `atoms_per_batch`, `memory_scales_with`) are rejected for these
  sources. Use a local **uncompressed** copy for those strategies.
- A structural scan does not turn a compressed/remote source into a seekable
  trajectory. ZIP uses seeking internally to read its central directory;
  remote ZIP uses obstore's reader for that purpose. This is not public
  random-access frame support. Tar enumeration and reading use separate passes,
  so a remote tar can be fetched more than once.

See [batching limits](batching-and-streaming.md) and the
[reader API](../reference/api.md) before choosing a large-data workflow.

## Object storage

Install the optional transport for S3 **and** Hub reads:

```sh title="Install the remote reader extra"
pip install "oxyz[s3]"
```

The extra supplies `obstore`; the base install raises an informative `ImportError`
for remote reads. `read`, `iread`, `scan`, `infer_schema`, batch readers and the
converter readers accept `storage_options=`. S3 credentials/region/endpoint use
obstore's provider configuration and `AWS_*` environment variables by default.
Explicit options can select a non-AWS S3-compatible service such as MinIO, R2
or Ceph. Provider options and HTTP client options share this dictionary; oxyz
routes recognised client options such as `timeout` and `default_headers`
separately to obstore. Use that provider's actual endpoint, region, credentials
and addressing requirements; there is no universal non-AWS configuration.

The following URLs and credentials are schematic, not available example data.

<!-- skip: next "schematic S3 object requires network access and provider credentials" -->

```python title="S3 and non-AWS endpoint syntax — schematic"
import oxyz

frames = oxyz.read("s3://bucket/train.extxyz.gz")
frames = oxyz.read(
    "s3://bucket/train.extxyz",
    storage_options={
        "endpoint": "https://minio.example",
        "region": "us-east-1",
        "access_key_id": "YOUR_ACCESS_KEY",
        "secret_access_key": "YOUR_SECRET_KEY",
        "virtual_hosted_style_request": False,
    },
)
```

Remote compression inference and archive `member=` selection follow the local
rules. A recognised key suffix avoids a magic-sniff request; otherwise a small
range request inspects the leading bytes. Remote ZIP needs server-side support
for the seek/range behaviour used by obstore.

`gs://` and `az://` (also `azure://`, `abfs://`, `abfss://`) route through obstore.
Google Cloud Storage and Azure are best-effort: oxyz's own integration tests
exercise S3 against a local mock service, not these providers. Their credentials
and configuration are provider-specific; S3 environment variables are not a
promise of authentication for them. No arbitrary HTTP filesystem is supported.

## HuggingFace Hub files

The syntax is `hf://[datasets|spaces|models/]owner/repo[@revision]/path`.
Without a kind prefix, the repo is a **model**; without `@revision`, the revision
is literally `main`. Specify a branch, tag or commit on the repo component,
for example `repo@v1.0/data/train.extxyz`. Explicit `models/` is accepted;
leading `datasets`/`spaces`/`models` segments are reserved as kind selectors,
not interpreted as owner names.

Only HTTP(S) URLs whose host is exactly `huggingface.co` or `hf.co` are claimed.
A browser file URL using `/blob/revision/path` is rewritten to
`/resolve/revision/path`, which serves file bytes. `/resolve/` is accepted;
an explicit `/raw/` remains `/raw/` (and may serve an LFS pointer rather than
large-file contents). Repo, directory and search pages are rejected as non-file
URLs; arbitrary HTTP(S) hosts are not remote inputs.

<!-- skip: next "schematic Hub files require network access and an existing repository" -->

```python title="Equivalent Hub file URLs and a pinned revision — schematic"
import oxyz

frames = oxyz.read("hf://datasets/owner/repo/train.extxyz")
frames = oxyz.read("https://huggingface.co/datasets/owner/repo/blob/main/train.extxyz")
pinned = oxyz.read("hf://datasets/owner/repo@v1.0/data/train.extxyz")
model_file = oxyz.read("hf://models/owner/repo/train.extxyz")
```

Private and gated repositories require permission and a token. An explicit
`default_headers` authorisation header wins over the environment, with a
case-insensitive header-name check. Otherwise the first non-empty value of
`HF_TOKEN`, then legacy `HUGGING_FACE_HUB_TOKEN`, becomes `Bearer <token>`.
No token is required merely to construct a public Hub request, and Hub tokens
are not attached to unrelated S3/provider requests.

<!-- skip: next "schematic private Hub file requires network access and an authorised token" -->

```python title="Explicit Hub authorisation — schematic"
import oxyz

frames = oxyz.read(
    "hf://datasets/owner/private/train.extxyz",
    storage_options={"default_headers": {"authorization": "Bearer hf_YOUR_TOKEN"}},
)
```

Hub Storage Buckets are a separate S3-compatible product, not Hub repository
files. Use an `s3://bucket/key` URL, an endpoint such as
`https://s3.hf.co/<namespace>`, and the bucket service's S3 credentials generated
from a Hub token. The `hf://` repo rewriting and automatic bearer header do not
configure those buckets. Endpoint and credential provisioning remain the
provider's responsibility; oxyz has no dedicated Storage Bucket integration test.

The [remote implementation](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_remote.py)
and [mocked remote tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_remote_unit.py)
cover URL rewriting, token precedence and dispatch without requiring live Hub
access. These skipped syntax examples do not demonstrate a network read.
