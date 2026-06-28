# Compressed fixtures

Compressed twins of `../two_frame_same_schema.xyz`, one per codec, plus a few
edge cases. Each must read back to the same two frames as the plain file.

Regenerate from `tests/data/` (set `COPYFILE_DISABLE=1` so macOS `tar` omits
`._` AppleDouble entries, which would otherwise look like extra members):

```sh
base=two_frame_same_schema.xyz
gzip -c -n "$base"        > compressed/two_frame.xyz.gz
zstd -q -c   "$base"      > compressed/two_frame.xyz.zst
zip  -q -j compressed/two_frame.xyz.zip "$base"
COPYFILE_DISABLE=1 tar -czf compressed/two_frame.tar.gz "$base"
COPYFILE_DISABLE=1 tar -cf  compressed/two_frame.tar    "$base"
cat compressed/two_frame.xyz.gz  compressed/two_frame.xyz.gz  > compressed/concat.xyz.gz
cat compressed/two_frame.xyz.zst compressed/two_frame.xyz.zst > compressed/concat.xyz.zst
cp "$base" a.xyz; cp varying_atom_counts.xyz b.xyz
zip -q -j compressed/multi_member.zip a.xyz b.xyz
COPYFILE_DISABLE=1 tar -czf compressed/multi_member.tar.gz a.xyz b.xyz
rm -f a.xyz b.xyz
```

| File | Exercises |
|---|---|
| `two_frame.xyz.gz` | gzip single stream (`flate2`/`MultiGzDecoder`). |
| `two_frame.xyz.zst` | zstd single stream (`ruzstd` streaming decode). |
| `two_frame.xyz.zip` | zip with one member. |
| `two_frame.tar.gz` | gzip-compressed tar with one member. |
| `two_frame.tar` | uncompressed tar with one member. |
| `concat.xyz.gz` | Two concatenated gzip members — `MultiGzDecoder` must read both (4 frames). |
| `concat.xyz.zst` | Two concatenated zstd frames — the multi-frame wrapper must read both (4 frames). |
| `multi_member.zip` | Two `.xyz` members (`a.xyz`, `b.xyz`) — ambiguous without `member=`. |
| `multi_member.tar.gz` | Same ambiguity for the tar path. |
