"""Flag out-of-distribution / low-quality episodes to exclude from training.

This scans every episode in a LeRobot tele-op dataset using only the
proprioceptive + action time series (no video), and reports two tiers:

  * INTEGRITY failures  -> high-confidence exclude list. Absolute, distribution-
    free checks: non-finite values, truncated (too few frames), or frozen
    (essentially no motion). These almost certainly should not be trained on.

  * STATISTICAL outliers -> ranked review list. Per-feature robust z-scores
    (median / MAD) over the whole dataset. "Badness" features (jerk, velocity
    variance, action entropy, command-tracking error) are treated one-sided:
    only unusually *high* values are suspicious. An unusually smooth/quiet demo
    is in-distribution and is NOT flagged. These are candidates to eyeball in
    the review GUI before deciding to drop them -- they are deliberately kept
    out of the confident exclude list.

Usage:
    python -m teleop_data_analyzer.check_ood \
        --dataset-root data/red_cube_cardbox_all_cleaned_01

    python -m teleop_data_analyzer.check_ood \
        --dataset-root data/red_cube_cardbox_all_cleaned_01 \
        --out ood_report --z-thresh 3.5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

try:
    from teleop_data_analyzer.metrics import (
        action_entropy,
        gripper_aperture,
        jerk,
        joint_velocity_variance,
    )
    from teleop_data_analyzer.sim.action_data import list_episodes, load_episode
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from teleop_data_analyzer.metrics import (
        action_entropy,
        gripper_aperture,
        jerk,
        joint_velocity_variance,
    )
    from teleop_data_analyzer.sim.action_data import list_episodes, load_episode


# Aggregate features used for statistical outlier detection.
# direction "high": only unusually large values are suspicious (one-sided).
# direction "both": either tail is suspicious (two-sided).
FEATURES: list[tuple[str, str, str]] = [
    # key,            direction, human label
    ("frames",          "both", "frame count"),
    ("jerk_mean",       "high", "mean jerk"),
    ("jerk_p95",        "high", "p95 jerk"),
    ("velvar_mean",     "high", "mean velocity variance"),
    ("velvar_p95",      "high", "p95 velocity variance"),
    ("entropy_mean",    "high", "mean action entropy"),
    ("track_err_mean",  "high", "mean command-tracking error"),
    ("track_err_p95",   "high", "p95 command-tracking error"),
    ("motion_per_frame", "high", "mean joint motion / frame"),
    ("gripper_range",   "both", "gripper aperture range"),
]
FEATURE_KEYS = [key for key, _, _ in FEATURES]
FEATURE_LABEL = {key: label for key, _, label in FEATURES}
FEATURE_DIR = {key: direction for key, direction, _ in FEATURES}

MAD_SCALE = 1.4826  # makes MAD a consistent estimator of std for normal data


@dataclass
class EpisodeResult:
    index: int
    frames: int
    fps: float
    task: str
    features: dict[str, float]
    integrity_reasons: list[str] = field(default_factory=list)
    # filled in after the dataset-wide statistics are known:
    zscores: dict[str, float] = field(default_factory=dict)
    triggered: list[dict] = field(default_factory=list)
    score: float = 0.0

    @property
    def integrity_fail(self) -> bool:
        return bool(self.integrity_reasons)

    @property
    def statistical_outlier(self) -> bool:
        return bool(self.triggered)


def _finite_fraction(*arrays: np.ndarray) -> float:
    """Fraction of non-finite (NaN/Inf) entries across the given arrays."""
    total = 0
    bad = 0
    for arr in arrays:
        if arr.size == 0:
            continue
        total += arr.size
        bad += int(np.count_nonzero(~np.isfinite(arr)))
    return bad / total if total else 0.0


def _p95(values: np.ndarray) -> float:
    return float(np.nanpercentile(values, 95)) if values.size else float("nan")


def extract_features(motion, entropy_window: int, entropy_bins: int) -> dict[str, float]:
    """Per-episode aggregate features from the action / state time series."""
    state = motion.state
    action = motion.action

    jerk_series = jerk(state, motion.fps)
    velvar_series = joint_velocity_variance(state, motion.fps)
    entropy_series = action_entropy(action, window=entropy_window, bins=entropy_bins)

    # command-tracking error: how far measured state was from commanded action.
    track_err = np.linalg.norm(action - state, axis=1)

    # average per-frame joint motion (used both as a feature and a frozen gate).
    if state.shape[0] >= 2:
        motion_per_frame = float(np.nanmean(np.abs(np.diff(state, axis=0))))
    else:
        motion_per_frame = 0.0

    try:
        left, right = gripper_aperture(state, motion.joint_names)
        gripper_range = float(max(np.ptp(left), np.ptp(right)))
    except ValueError:
        gripper_range = float("nan")

    return {
        "frames": float(motion.num_frames),
        "jerk_mean": float(np.nanmean(jerk_series)),
        "jerk_p95": _p95(jerk_series),
        "velvar_mean": float(np.nanmean(velvar_series)),
        "velvar_p95": _p95(velvar_series),
        "entropy_mean": float(np.nanmean(entropy_series)),
        "track_err_mean": float(np.nanmean(track_err)),
        "track_err_p95": _p95(track_err),
        "motion_per_frame": motion_per_frame,
        "gripper_range": gripper_range,
    }


def load_episode_result(
    dataset_root: str,
    episode_index: int,
    entropy_window: int,
    entropy_bins: int,
    min_frames: int,
    frozen_eps: float,
) -> EpisodeResult:
    """Load one episode, compute features, and apply the hard integrity gates."""
    motion = load_episode(dataset_root, episode_index)
    features = extract_features(motion, entropy_window, entropy_bins)

    reasons: list[str] = []
    nan_frac = _finite_fraction(motion.action, motion.state)
    if nan_frac > 0:
        reasons.append(f"non_finite_values ({nan_frac:.1%} of entries)")
    if motion.num_frames < min_frames:
        reasons.append(f"too_short ({motion.num_frames} < {min_frames} frames)")
    if features["motion_per_frame"] < frozen_eps:
        reasons.append(
            f"frozen (motion/frame {features['motion_per_frame']:.2e} < {frozen_eps:.1e})"
        )

    return EpisodeResult(
        index=motion.index,
        frames=motion.num_frames,
        fps=float(motion.fps),
        task=motion.task,
        features=features,
        integrity_reasons=reasons,
    )


def robust_zscores(values: np.ndarray) -> np.ndarray:
    """Signed robust z-scores via median / MAD, falling back to std.

    NaNs are ignored when estimating the center/scale and pass through as NaN.
    A constant feature (zero spread) yields all-zero z-scores.
    """
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scale = MAD_SCALE * mad
    if not np.isfinite(scale) or scale == 0:
        std = np.nanstd(values)
        scale = std if (np.isfinite(std) and std > 0) else np.nan
    if not np.isfinite(scale):
        return np.zeros_like(values)
    return (values - median) / scale


def score_outliers(results: list[EpisodeResult], z_thresh: float) -> None:
    """Annotate each result with per-feature z-scores and a suspicion score."""
    n = len(results)
    matrix = {
        key: np.array([r.features.get(key, np.nan) for r in results], dtype=np.float64)
        for key in FEATURE_KEYS
    }
    zmatrix = {key: robust_zscores(col) for key, col in matrix.items()}

    for i, r in enumerate(results):
        triggered: list[dict] = []
        score = 0.0
        for key in FEATURE_KEYS:
            z = float(zmatrix[key][i])
            r.zscores[key] = z
            if not np.isfinite(z):
                continue
            # one-sided "high" features only care about the upper tail.
            suspicion = abs(z) if FEATURE_DIR[key] == "both" else max(z, 0.0)
            score = max(score, suspicion)
            if suspicion > z_thresh:
                triggered.append(
                    {
                        "feature": key,
                        "label": FEATURE_LABEL[key],
                        "value": float(r.features.get(key, np.nan)),
                        "z": z,
                    }
                )
        r.triggered = sorted(triggered, key=lambda t: -abs(t["z"]))
        r.score = float(score)


def build_manifest(
    dataset_root: str,
    results: list[EpisodeResult],
    params: dict,
) -> dict:
    integrity = [r for r in results if r.integrity_fail]
    review = sorted(
        (r for r in results if r.statistical_outlier and not r.integrity_fail),
        key=lambda r: -r.score,
    )
    return {
        "dataset_root": dataset_root,
        "n_episodes": len(results),
        "params": params,
        "n_excluded": len(integrity),
        "n_review": len(review),
        "small_sample_warning": len(results) < 20,
        "exclude": [
            {
                "index": r.index,
                "frames": r.frames,
                "reasons": r.integrity_reasons,
            }
            for r in sorted(integrity, key=lambda r: r.index)
        ],
        "review": [
            {
                "index": r.index,
                "frames": r.frames,
                "score": round(r.score, 3),
                "triggered": [
                    {"feature": t["feature"], "value": round(t["value"], 5), "z": round(t["z"], 3)}
                    for t in r.triggered
                ],
            }
            for r in review
        ],
        "exclude_indices": sorted(r.index for r in integrity),
        "review_indices": [r.index for r in review],
    }


def write_csv(path: Path, results: list[EpisodeResult]) -> None:
    # "frames" is already one of FEATURE_KEYS, so it is not repeated in the prefix.
    fieldnames = (
        ["index", "fps", "duration_s"]
        + FEATURE_KEYS
        + [f"z_{key}" for key in FEATURE_KEYS]
        + ["integrity_fail", "statistical_outlier", "score", "reasons", "task"]
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda r: r.index):
            row = {
                "index": r.index,
                "fps": f"{r.fps:g}",
                "duration_s": round(r.frames / r.fps, 3) if r.fps else "",
                "integrity_fail": int(r.integrity_fail),
                "statistical_outlier": int(r.statistical_outlier),
                "score": round(r.score, 3),
                "reasons": "; ".join(
                    r.integrity_reasons
                    + [f"{t['feature']}(z={t['z']:+.1f})" for t in r.triggered]
                ),
                "task": r.task,
            }
            for key in FEATURE_KEYS:
                row[key] = round(r.features.get(key, float("nan")), 5)
                row[f"z_{key}"] = round(r.zscores.get(key, float("nan")), 3)
            writer.writerow(row)


def print_report(results: list[EpisodeResult], manifest: dict, top: int) -> None:
    n = len(results)
    print(f"\n=== OOD check: {n} episodes ===")
    if manifest["small_sample_warning"]:
        print(
            "  ! WARNING: fewer than 20 episodes -- robust z-scores are noisy; "
            "treat the statistical tier as low-confidence."
        )

    integrity = [r for r in results if r.integrity_fail]
    print(f"\n-- Integrity failures (exclude): {len(integrity)} --")
    if not integrity:
        print("  none")
    for r in sorted(integrity, key=lambda r: r.index):
        print(f"  episode {r.index:>4}: {', '.join(r.integrity_reasons)}")

    review = [r for r in results if r.statistical_outlier and not r.integrity_fail]
    review.sort(key=lambda r: -r.score)
    print(f"\n-- Statistical outliers (review, ranked): {len(review)} --")
    if not review:
        print("  none")
    for r in review[:top]:
        triggers = ", ".join(
            f"{t['label']}={t['value']:.3g} (z={t['z']:+.1f})" for t in r.triggered
        )
        print(f"  episode {r.index:>4}  score={r.score:5.1f}  {triggers}")
    if len(review) > top:
        print(f"  ... and {len(review) - top} more (see CSV)")


def run(args: argparse.Namespace) -> int:
    dataset_root = str(Path(args.dataset_root).expanduser())
    paths = list_episodes(dataset_root)
    if args.limit:
        paths = paths[: args.limit]
    print(f"Scanning {len(paths)} episodes under {dataset_root}")

    results: list[EpisodeResult] = []
    for offset, path in enumerate(paths):
        stem = Path(path).stem  # episode_000123
        try:
            episode_index = int(stem.split("_")[-1])
        except ValueError:
            episode_index = offset
        try:
            result = load_episode_result(
                dataset_root,
                episode_index,
                args.entropy_window,
                args.entropy_bins,
                args.min_frames,
                args.frozen_eps,
            )
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't kill the run
            print(f"  episode {episode_index}: LOAD ERROR -> excluded ({exc})")
            results.append(
                EpisodeResult(
                    index=episode_index,
                    frames=0,
                    fps=0.0,
                    task="",
                    features={key: float("nan") for key in FEATURE_KEYS},
                    integrity_reasons=[f"load_error ({type(exc).__name__})"],
                )
            )
            continue
        results.append(result)
        if (offset + 1) % 25 == 0 or offset + 1 == len(paths):
            print(f"  ...{offset + 1}/{len(paths)}")

    score_outliers(results, args.z_thresh)

    params = {
        "z_thresh": args.z_thresh,
        "min_frames": args.min_frames,
        "frozen_eps": args.frozen_eps,
        "entropy_window": args.entropy_window,
        "entropy_bins": args.entropy_bins,
    }
    manifest = build_manifest(dataset_root, results, params)
    print_report(results, manifest, top=args.top)

    if not args.no_write:
        out_dir = Path(args.out).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.json"
        csv_path = out_dir / "episodes.csv"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        write_csv(csv_path, results)
        print(f"\nWrote {manifest_path}")
        print(f"Wrote {csv_path}")
        print(
            f"\nExclude {manifest['n_excluded']} integrity failures; "
            f"review {manifest['n_review']} statistical outliers before dropping."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, help="LeRobot dataset root")
    parser.add_argument("--out", default="ood_report", help="output dir for manifest.json + episodes.csv")
    parser.add_argument("--z-thresh", type=float, default=3.5, help="robust z-score flag threshold")
    # Defaults are calibrated to the G1 whole-body pick/place distribution, where
    # real demos run ~230-2600 frames with ~1e-3..3e-3 rad/frame of joint motion.
    # Both thresholds sit in a clear gap below the normal range (frame counts jump
    # 80 -> 229; motion jumps 5.9e-4 -> 1.0e-3), so they catch aborted/near-frozen
    # clips without touching genuine demos. Tune per dataset.
    parser.add_argument("--min-frames", type=int, default=100, help="episodes shorter than this (~2s @ 50fps) are excluded")
    parser.add_argument(
        "--frozen-eps",
        type=float,
        default=8e-4,
        help="mean per-frame joint motion (rad) below which an episode is 'frozen' (barely moved)",
    )
    parser.add_argument("--entropy-window", type=int, default=16, help="action-entropy window")
    parser.add_argument("--entropy-bins", type=int, default=16, help="action-entropy histogram bins")
    parser.add_argument("--top", type=int, default=25, help="how many statistical outliers to print")
    parser.add_argument("--limit", type=int, help="only scan the first N episodes (debugging)")
    parser.add_argument("--no-write", action="store_true", help="print the report only; write no files")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
