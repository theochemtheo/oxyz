# /// script
# requires-python = ">=3.12"
# dependencies = ["pypi-attestations==0.0.31"]
# ///
"""Convert an ``actions/attest`` SLSA provenance bundle into a PyPI attestation."""

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
    parser.add_argument("bundle", type=Path, help="Sigstore bundle from actions/attest")
    parser.add_argument("dist", type=Path, help="the wheel or sdist it attests")
    parser.add_argument("--repository", required=True, help="owner/repo")
    parser.add_argument("--workflow", required=True, help="workflow file name")
    args = parser.parse_args()

    attestation = Attestation.from_bundle(Bundle.from_json(args.bundle.read_bytes()))
    publisher = GitHubPublisher(repository=args.repository, workflow=args.workflow)
    dist = Distribution.from_file(args.dist)
    predicate_type, _ = attestation.verify(publisher, dist)
    if predicate_type != AttestationType.SLSA_PROVENANCE_V1:
        raise SystemExit(f"{args.bundle}: not SLSA provenance: {predicate_type}")

    target = args.dist.with_name(f"{args.dist.name}.slsa.attestation")
    target.write_text(attestation.model_dump_json(), encoding="utf-8")


if __name__ == "__main__":
    main()
