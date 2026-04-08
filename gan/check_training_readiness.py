import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPORTS_ROOT = os.path.join(PROJECT_ROOT, "reports", "gan_validation")
DATASETS = ["gold_RRL_interpolate", "silver_RRL_interpolate"]


def evaluate_dataset(dataset_name):
    metrics_path = os.path.join(REPORTS_ROOT, dataset_name, "selected_candidate_metrics.json")
    if not os.path.exists(metrics_path):
        return {"dataset": dataset_name, "status": "missing", "reasons": ["missing selected_candidate_metrics.json"]}

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    reasons = []
    if metrics.get("quality_label") == "reject":
        reasons.append(f"quality_label={metrics['quality_label']}")
    if metrics.get("avg_ks_stat", 1.0) > 0.12:
        reasons.append(f"avg_ks_stat={metrics['avg_ks_stat']:.4f} > 0.12")
    if metrics.get("avg_acf_gap", 1.0) > 0.15:
        reasons.append(f"avg_acf_gap={metrics['avg_acf_gap']:.4f} > 0.15")
    if metrics.get("corr_gap", 1.0) > 0.25:
        reasons.append(f"corr_gap={metrics['corr_gap']:.4f} > 0.25")

    return {
        "dataset": dataset_name,
        "status": "ready" if not reasons else "not_ready",
        "reasons": reasons,
        "metrics": metrics,
    }


def main():
    overall_ready = True
    for dataset_name in DATASETS:
        result = evaluate_dataset(dataset_name)
        print(f"\n[{dataset_name}] {result['status'].upper()}")
        if "metrics" in result:
            metrics = result["metrics"]
            print(f"  quality_label: {metrics.get('quality_label')}")
            print(f"  avg_ks_stat: {metrics.get('avg_ks_stat'):.4f}")
            print(f"  avg_acf_gap: {metrics.get('avg_acf_gap'):.4f}")
            print(f"  corr_gap: {metrics.get('corr_gap'):.4f}")
        if result["reasons"]:
            overall_ready = False
            for reason in result["reasons"]:
                print(f"  reason: {reason}")

    sys.exit(0 if overall_ready else 1)


if __name__ == "__main__":
    main()
