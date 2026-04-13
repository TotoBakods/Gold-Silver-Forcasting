"""
Quick fix: Reset the gold diagnostic_logs in simulation_state.json.

The 50 buffered entries all come from the stuck-flush period (100% is_flush=True,
systematic ~2-3% over-prediction). With those in the buffer, R² will never recover
because the SSE calculation is permanently biased.

This script:
1. Clears gold diagnostic_logs
2. Resets gold seed_errors
3. Leaves history (price chart) and current_date/test_idx intact so the
   visual chart still shows the full simulation run.
"""
import json

STATE_FILE = "simulation_state.json"

with open(STATE_FILE, "r", encoding="utf-8") as f:
    state = json.load(f)

gold = state["gold"]

print(f"Before: {len(gold['diagnostic_logs'])} diagnostic log entries for gold")
print(f"Gold current_date: {gold['current_date']}, test_idx: {gold['test_idx']}")

# ── Clear stale diagnostic buffer ─────────────────────────────────────────────
gold["diagnostic_logs"] = []

# ── Reset seed errors (so adaptive weights start fresh) ───────────────────────
gold["seed_errors"] = {k: [] for k in gold["seed_errors"]}

print(f"After:  {len(gold['diagnostic_logs'])} diagnostic log entries for gold")
print("Reset complete. Diagnostic buffer cleared. History/dates preserved.")
print("Restart the API server for changes to take effect.")

with open(STATE_FILE, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)

print(f"Saved to {STATE_FILE}")
