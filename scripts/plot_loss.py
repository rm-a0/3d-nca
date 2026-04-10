import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _phase_transitions(df: pd.DataFrame) -> pd.DataFrame:
    if "phase" not in df.columns:
        return pd.DataFrame(columns=["epoch", "phase"])
    phase = df["phase"].astype("string").str.strip()
    phase = phase.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # Ignore missing labels so empty phase columns do not create fake transitions.
    changed = phase.ne(phase.shift(1)).fillna(False) & phase.notna()
    changed &= df.index.to_series().ne(df.index[0])

    transitions = df.loc[changed, ["epoch"]].copy()
    transitions["phase"] = phase.loc[changed].astype(str)
    return transitions


def _curriculum_steps(epochs: np.ndarray) -> np.ndarray:
    ramp = 8.0 + (32.0 - 8.0) * (epochs / 2000.0)
    return np.where(epochs <= 2000.0, ramp, 32.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NCA losses with curriculum ramp")
    parser.add_argument("--run", default="001", help="Run id (default: 001)")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--base-dir", default="runs")
    args = parser.parse_args()

    run_dir = Path(args.base_dir) / f"run_{args.run}"
    csv_path = run_dir / "loss.csv"
    out_path = Path(args.out) if args.out else run_dir / "plot_loss.png"

    df = pd.read_csv(csv_path)
    epochs = df["epoch"].to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    line_alpha, = ax1.plot(epochs, df["loss_alpha"], label="loss_alpha", color="#1f77b4")
    line_color, = ax1.plot(epochs, df["loss_color"], label="loss_color", color="#ff7f0e")
    line_overflow, = ax1.plot(epochs, df["loss_overflow"], label="loss_overflow", color="#2ca02c")
    line_total, = ax1.plot(epochs, df["loss_total"], label="loss_total", color="#d62728")

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    curriculum = _curriculum_steps(epochs)
    line_curriculum, = ax2.plot(
        epochs,
        curriculum,
        "k--",
        alpha=0.3,
        label="curriculum steps",
    )
    ax2.set_ylabel("curriculum steps")

    transitions = _phase_transitions(df)
    for _, row in transitions.iterrows():
        ep = float(row["epoch"])
        phase = str(row["phase"])
        ax1.axvline(ep, color="grey", linestyle="--", alpha=0.6)
        ax1.text(
            ep,
            0.97,
            phase,
            transform=ax1.get_xaxis_transform(),
            ha="left",
            va="top",
            color="grey",
            fontsize=8,
        )

    handles = [line_alpha, line_color, line_overflow, line_total, line_curriculum]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
