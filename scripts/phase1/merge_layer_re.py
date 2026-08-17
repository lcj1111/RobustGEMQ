import argparse
import math
import pickle
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge expert-sharded layer reconstruction errors")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--bits", default="1,2,3")
    return parser.parse_args()


def main():
    args = parse_args()
    expected_bits = {int(bit) for bit in args.bits.split(",")}
    merged = {layer: {} for layer in range(args.layers)}

    for path in args.inputs:
        with path.open("rb") as handle:
            shard = pickle.load(handle)
        if set(shard) != set(merged):
            raise ValueError(f"{path} has layers {sorted(shard)}, expected 0..{args.layers - 1}")
        for layer, experts in shard.items():
            overlap = set(merged[layer]).intersection(experts)
            if overlap:
                raise ValueError(f"{path} duplicates layer {layer} experts {sorted(overlap)}")
            for expert, values in experts.items():
                if set(values) != expected_bits:
                    raise ValueError(
                        f"{path}: layer {layer} expert {expert} has bits {sorted(values)}, "
                        f"expected {sorted(expected_bits)}"
                    )
                for bit, value in values.items():
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"{path}: layer {layer} expert {expert} bit {bit} "
                            f"has non-numeric coefficient {value!r}"
                        ) from error
                    if not math.isfinite(numeric) or numeric < 0:
                        raise ValueError(
                            f"{path}: layer {layer} expert {expert} bit {bit} "
                            f"has invalid coefficient {value!r}"
                        )
                merged[layer][expert] = values

    expected_experts = set(range(args.experts))
    for layer, experts in merged.items():
        if set(experts) != expected_experts:
            missing = sorted(expected_experts - set(experts))
            extra = sorted(set(experts) - expected_experts)
            raise ValueError(f"layer {layer}: missing={missing}, extra={extra}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(merged, handle)
    print(
        f"merged {len(args.inputs)} shards -> {args.output} "
        f"({args.layers} layers x {args.experts} experts x {len(expected_bits)} bits)"
    )


if __name__ == "__main__":
    main()
