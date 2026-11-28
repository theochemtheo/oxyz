# /// script
# requires-python = ">=3.12"
# dependencies = ["pypi-attestations==0.0.31"]
# ///
"""Convert ``actions/attest`` SLSA provenance bundles into PyPI attestations.

Every wheel and sdist in ``--dist-dir`` must be the sole subject of exactly one
bundle; anything unmatched is an error, so no file is published without
provenance.
"""

import argparse
from pathlib import Path

from pypi_attestations import (
    Attestation,
    AttestationType,
    Distribution,
    GitHubPublisher,
)
from sigstore.models import Bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundles", nargs="+", type=Path, help="Sigstore bundles from actions/attest"
    )
    parser.add_argument(
        "--dist-dir",
        required=True,
        type=Path,
        help="directory holding every wheel and sdist to publish",
    )
    parser.add_argument("--repository", required=True, help="owner/repo")
    parser.add_argument("--workflow", required=True, help="workflow file name")
    args = parser.parse_args()

    dists = {
        path.name: path
        for path in args.dist_dir.iterdir()
        if path.name.endswith((".whl", ".tar.gz"))
    }
    publisher = GitHubPublisher(repository=args.repository, workflow=args.workflow)
    prepared: dict[str, tuple[Path, Attestation]] = {}

    for bundle in args.bundles:
        attestation = Attestation.from_bundle(Bundle.from_json(bundle.read_bytes()))
        subjects = attestation.statement.get("subject", [])
        if len(subjects) != 1:
            raise SystemExit(
                f"{bundle}: {len(subjects)} subjects; PyPI needs exactly one"
            )
        name = subjects[0].get("name", "")
        if name not in dists:
            raise SystemExit(f"{bundle}: subject {name!r} is not in {args.dist_dir}")
        if name in prepared:
            raise SystemExit(
                f"{bundle}: {name} is already attested by {prepared[name][0]}"
            )

        predicate_type, _ = attestation.verify(
            publisher, Distribution.from_file(dists[name])
        )
        if predicate_type != AttestationType.SLSA_PROVENANCE_V1:
            raise SystemExit(f"{bundle}: not SLSA provenance: {predicate_type}")
        prepared[name] = (bundle, attestation)

    if missing := sorted(dists.keys() - prepared.keys()):
        raise SystemExit(f"no provenance for: {', '.join(missing)}")

    for name, (_, attestation) in prepared.items():
        target = dists[name].with_name(f"{name}.slsa.attestation")
        target.write_text(attestation.model_dump_json(), encoding="utf-8")


if __name__ == "__main__":
    main()
