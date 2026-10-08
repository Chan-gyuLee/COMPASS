"""Command-line interface. Use `compass-sample --help` after installation."""

import argparse
import json
import logging
import os
from pathlib import Path

from .config import SteeringConfig


def parser():
    result = argparse.ArgumentParser(
        description="Steer BioEmu by optimizing conditioning embeddings."
    )
    inputs = result.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sequence", help="Protein sequence (20 standard amino acids)")
    inputs.add_argument(
        "--fasta", type=Path, help="FASTA containing exactly one sequence"
    )
    result.add_argument(
        "--target", type=Path, required=True, help="Guidance PDB or mmCIF"
    )
    result.add_argument("--chain", required=True, help="Target chain identifier")
    result.add_argument(
        "--output", type=Path, required=True, help="New or empty output directory"
    )
    result.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "compass",
    )
    result.add_argument(
        "--single-embeddings", type=Path, help="Explicit single embeddings .npy"
    )
    result.add_argument(
        "--pair-embeddings", type=Path, help="Explicit pair embeddings .npy"
    )
    result.add_argument(
        "--config", type=Path, help="JSON object of SteeringConfig fields"
    )
    result.add_argument("--num-samples", type=int, default=1)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", default="cuda")
    result.add_argument("--model", default="bioemu-v1.1")
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate sequence, target and config without loading model or embeddings",
    )
    return result


def main(argv=None):
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    from Bio import SeqIO
    from .target import load_target, validate_sequence

    try:
        if args.fasta:
            records = list(SeqIO.parse(args.fasta, "fasta"))
            if len(records) != 1:
                raise ValueError("FASTA must contain exactly one sequence")
            sequence = str(records[0].seq)
        else:
            sequence = args.sequence
        sequence = validate_sequence(sequence)
        cfg = (
            SteeringConfig(**json.loads(args.config.read_text()))
            if args.config
            else SteeringConfig()
        )
        if args.num_samples < 1 or args.batch_size < 1 or not 0 <= args.seed < 2**32:
            raise ValueError(
                "Sample counts must be positive and seed must be in [0, 2**32)"
            )
        if bool(args.single_embeddings) != bool(args.pair_embeddings):
            raise ValueError("Provide both --single-embeddings and --pair-embeddings")
        for path in (args.single_embeddings, args.pair_embeddings):
            if path is not None and not path.is_file():
                raise ValueError(f"Embedding file not found: {path}")
        if args.output.exists() and (
            not args.output.is_dir() or any(args.output.iterdir())
        ):
            raise ValueError(f"Output must be a new or empty directory: {args.output}")
        target = load_target(args.target, args.chain, sequence)
        if args.dry_run:
            print(json.dumps(target.metadata(), indent=2))
            return 0
    except (OSError, ValueError, TypeError) as error:
        arg_parser.error(str(error))
    from .pipeline import CompassPipeline, seed_everything

    seed_everything(args.seed)
    args.cache_dir = args.cache_dir.expanduser()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    pipeline = CompassPipeline.from_pretrained(
        cfg,
        device=args.device,
        model_name=args.model,
        cache_dir=args.cache_dir,
    )
    pipeline.run(
        sequence=sequence,
        target_path=args.target,
        chain_id=args.chain,
        output_dir=args.output,
        cache_dir=args.cache_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        single_embeddings=args.single_embeddings,
        pair_embeddings=args.pair_embeddings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
