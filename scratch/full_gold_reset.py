"""
Full gold simulation reset:
- Clears diagnostic_logs
- Clears history  
- Resets test_idx to 0
- Resets current_date to first test date
- Resets seed_errors

This forces the API to rebuild model weights from base on restart,
then retrain fresh with the corrected loss function (calibrated thresholds).
"""
import json

STATE_FILE = "simulation_state.json"

with open(STATE_FILE, "r", encoding="utf-8") as f:
    state = json.load(f)

gold = state["gold"]

print(f"Before: test_idx={gold['test_idx']}, current_date={gold['current_date']}")
print(f"        {len(gold['diagnostic_logs'])} diag logs, {len(gold['history'])} history points")

gold["diagnostic_logs"] = []
gold["history"] = {}
gold["seed_errors"] = {k: [] for k in gold["seed_errors"]}
# Reset to day 0 — API will load base model weights on startup
gold["test_idx"] = 0
gold["current_date"] = "2026-04-10"  # first test date

print(f"After:  test_idx={gold['test_idx']}, current_date={gold['current_date']}")
print(f"        {len(gold['diagnostic_logs'])} diag logs, {len(gold['history'])} history points")
print("Full reset complete. Restart API server and simulation will begin fresh.")

with open(STATE_FILE, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)

print(f"Saved to {STATE_FILE}")
