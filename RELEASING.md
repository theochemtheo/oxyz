# Releasing

A release is cut by pushing a version tag. The `wheels.yml` workflow then
builds the wheels and sdist, publishes them to PyPI via trusted publishing with
SLSA provenance and a publish attestation for each file, and creates the GitHub
Release — notes taken from this version's `CHANGELOG.md` section, with the built
distributions attached. The tag is the only manual trigger; publishing requires
approval in the `pypi` environment.

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

The `pypi` environment allows only `v*` tags, requires approval from
`theochemtheo`, and disallows administrator bypass. The `release-tags` ruleset
restricts creation, updates, and deletion of `v*` tags to `theochemtheo`.

## Attestations and permissions

Build jobs have only `contents: read`. Fresh signing jobs download each build's
artifact from the same workflow run and generate one SLSA bundle per file,
without executing project code or artifact contents. An unprivileged job
converts the bundles with `scripts/pypi_slsa_attestation.py`, whose dependencies
are locked in its adjacent `.lock` file. Conversion verifies the signature,
repository, workflow, filename, digest, and single-subject SLSA statement before
preparing anything for publication.

The protected PyPI job only downloads those prepared files and runs
`pypa/gh-action-pypi-publish`, which generates publish attestations and uploads
both attestation types. It has `id-token: write`, but no repository write
permission or checkout. The GitHub Release job has `contents: write`, but no
OIDC permission.

SLSA provenance identifies the workflow run and distribution hash; it does not
prove that build dependencies are harmless or claim SLSA Build Level 3.

## Versioning

oxyz follows [Semantic Versioning](https://semver.org): within a major version
no release removes or incompatibly changes a public name, though new ones may be
added; a breaking change bumps the major. Every change is recorded in
`CHANGELOG.md`.
