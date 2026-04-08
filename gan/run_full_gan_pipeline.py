import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPORTS_ROOT = PROJECT_ROOT / "reports" / "gan_validation"
STRICT_READINESS = os.getenv("GAN_STRICT_READINESS", "1") != "0"

GENERATED_FILES = [
    PROJECT_ROOT / "gold_RRL_interpolate_extended.csv",
    PROJECT_ROOT / "silver_RRL_interpolate_extended.csv",
    SCRIPT_DIR / "gold_RRL_interpolate_stationary_gen.pth",
    SCRIPT_DIR / "silver_RRL_interpolate_stationary_gen.pth",
    SCRIPT_DIR / "gold_RRL_interpolate_stationary_path.png",
    SCRIPT_DIR / "silver_RRL_interpolate_stationary_path.png",
    SCRIPT_DIR / "gan_training_stationary.log",
]
DEFAULT_BATCH_CANDIDATES = [512, 384, 320, 256, 192, 128, 96, 64, 48, 32]


def remove_path(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
        print(f"Removed directory: {path}", flush=True)
    elif path.exists():
        path.unlink()
        print(f"Removed file: {path}", flush=True)


def clean_outputs():
    print("Cleaning previous GAN outputs...", flush=True)
    for path in GENERATED_FILES:
        remove_path(path)
    if REPORTS_ROOT.exists():
        remove_path(REPORTS_ROOT)


def run_step(step_name: str, script_name: str, extra_env: dict | None = None):
    print(f"\n=== {step_name} ===", flush=True)
    cmd = [sys.executable, str(SCRIPT_DIR / script_name)]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd)


def oom_detected() -> bool:
    log_path = SCRIPT_DIR / "gan_training_stationary.log"
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    markers = [
        "out of memory",
        "cuda error: out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
    ]
    return any(marker in text for marker in markers)


def candidate_batch_sizes() -> list[int]:
    env_value = os.getenv("GAN_BATCH_SIZE_CANDIDATES")
    if env_value:
        parsed = []
        for item in env_value.split(","):
            item = item.strip()
            if item:
                parsed.append(int(item))
        if parsed:
            return parsed

    explicit_batch = os.getenv("GAN_BATCH_SIZE")
    if explicit_batch:
        start = int(explicit_batch)
        candidates = [start]
        for size in DEFAULT_BATCH_CANDIDATES:
            if size < start:
                candidates.append(size)
        return candidates

    return DEFAULT_BATCH_CANDIDATES


def run_training_with_batch_fallback():
    last_error = None
    for batch_size in candidate_batch_sizes():
        clean_outputs()
        print(f"\nTrying training with GAN_BATCH_SIZE={batch_size}", flush=True)
        try:
            run_step(
                f"Train And Generate Data (batch_size={batch_size})",
                "generate_gan_data.py",
                extra_env={"GAN_BATCH_SIZE": str(batch_size)},
            )
            return batch_size
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if oom_detected():
                print(f"CUDA OOM detected at batch size {batch_size}. Retrying smaller batch...", flush=True)
                continue
            raise SystemExit(f"Train And Generate Data failed with exit code {exc.returncode}") from exc

    raise SystemExit(
        f"Train And Generate Data failed for all batch sizes. Last error: {last_error.returncode if last_error else 'unknown'}"
    )


def main():
    used_batch_size = run_training_with_batch_fallback()
    run_step("Validate Generated Data", "validate_generated_data.py")
    run_step("Build GAN Validity Reports", "report_gan_validity.py")
    readiness_ok = True
    try:
        run_step("Check Training Readiness", "check_training_readiness.py")
    except subprocess.CalledProcessError:
        readiness_ok = False
        if STRICT_READINESS:
            print("\nTraining readiness check failed.", flush=True)
            print("Outputs were generated, but at least one dataset is not ready for forecasting-model training.", flush=True)
            print("Set GAN_STRICT_READINESS=0 if you want the pipeline to finish without failing the command.", flush=True)
            raise SystemExit(1)
        print("\nTraining readiness check reported issues, but strict mode is off so the pipeline will finish.", flush=True)
    print("\nGAN pipeline completed successfully.", flush=True)
    print(f"Used GAN batch size: {used_batch_size}", flush=True)
    print(f"Extended datasets: {PROJECT_ROOT / 'gold_RRL_interpolate_extended.csv'}", flush=True)
    print(f"Extended datasets: {PROJECT_ROOT / 'silver_RRL_interpolate_extended.csv'}", flush=True)
    print(f"Reports: {REPORTS_ROOT}", flush=True)
    if not readiness_ok:
        print("Readiness status: not all datasets are ready for training.", flush=True)


if __name__ == "__main__":
    main()
