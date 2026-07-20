# Releasing

A release is cut by pushing a version tag. The `wheels.yml` workflow then
builds the wheels and sdist, publishes them to PyPI via trusted publishing, and
creates the GitHub Release — notes taken from this version's `CHANGELOG.md`
section, with the built artifacts attached. The tag is the only manual trigger.

## Steps

1. Open a release PR that:
   - bumps the version in `pyproject.toml` and both `crates/*/Cargo.toml` (a
     build refreshes `Cargo.lock` and `uv.lock`);
   - adds a `CHANGELOG.md` section under a `## [X.Y.Z] - YYYY-MM-DD` heading,
     with a matching link reference at the foot of the file;
   - refreshes `benchmarks/RESULTS.md` if performance changed.
2. Preview the release notes: `scripts/changelog-section.sh X.Y.Z`.
3. Merge the PR.
4. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.

## Prerequisites

PyPI trusted publishing must be configured for the `oxyz` project — owner
`theochemtheo`, repository `oxyz`, workflow `wheels.yml`, environment `pypi`.

## Versioning

oxyz follows [Semantic Versioning](https://semver.org): within a major version
no release removes or incompatibly changes a public name, though new ones may be
added; a breaking change bumps the major. Every change is recorded in
`CHANGELOG.md`.
