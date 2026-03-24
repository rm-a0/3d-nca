import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def _phase_transitions(df: pd.DataFrame) -> pd.DataFrame:
    phase_str = df["phase"].astype(str)
    changed = phase_str.ne(phase_str.shift(1))
    return df.loc[changed & df.index.to_series().ne(df.index[0]), ["epoch", "phase"]]


def _read_events(path: Path) -> list[dict]:
    events: list[dict] = []
    if not path.exists():
        return events
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        return []
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NCA total loss with phase markers")
    parser.add_argument("--run", default="001", help="Run id (default: 001)")
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    run_dir = Path("runs") / f"run_{args.run}"
    csv_path = run_dir / "loss.csv"
    events_path = run_dir / "events.jsonl"
    out_path = Path(args.out) if args.out else run_dir / "plot_loss_phased.png"

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["epoch"], df["loss_total"], color="#1f77b4", label="loss_total")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss_total")
    ax.set_yscale("log")

    transitions = _phase_transitions(df)
    for _, row in transitions.iterrows():
        ep = float(row["epoch"])
        phase = str(row["phase"])
        ax.axvline(ep, color="red", linestyle="--", alpha=0.6)
        ax.text(
            ep,
            0.98,
            f"-> Phase {phase}",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            color="red",
            fontsize=8,
        )

    events = _read_events(events_path)
    for i, event in enumerate(events):
        epoch = event.get("epoch")
        event_type = event.get("event_type", "event")
        if epoch is None:
            continue
        y_pos = 0.90 - (i % 8) * 0.04
        ax.text(
            float(epoch),
            y_pos,
            str(event_type),
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=7,
            color="black",
            alpha=0.75,
        )

    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
