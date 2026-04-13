import json, numpy as np

logs = json.load(open('simulation_state.json'))['gold']['diagnostic_logs']
aa = np.array([l['actual_ret'] for l in logs])
ap = np.array([l['pred_ret'] for l in logs])

print(f"Total logs: {len(logs)}")
print(f"pred std:   {np.std(ap):.6f}  (should be > 0.005 to avoid laziness)")
print(f"actual std: {np.std(aa):.6f}")
print(f"pred range: [{ap.min():.5f}, {ap.max():.5f}]")
print()

# Rolling Pearson r2 for different windows
print("=== Rolling Pearson r2 by window ===")
for w in [10, 15, 20, 30, 50]:
    a = aa[-w:]; p = ap[-w:]
    if np.std(a) > 1e-9 and np.std(p) > 1e-9:
        r = np.corrcoef(a, p)[0,1]
        rmse = np.sqrt(np.mean((a - p)**2))
        print(f'Last {w:2d} days: r={r:+.3f}  r2={np.sign(r)*r**2:+.4f}  dir_acc={np.mean(np.sign(p)==np.sign(a)):.0%}  pred_std={np.std(p):.5f}  rmse={rmse:.5f}')
    else:
        print(f'Last {w:2d} days: ZERO pred variance — model is lazy!')

print()
print("=== Last 20 days sign tracking ===")
for l in logs[-20:]:
    a_sign = '+' if l['actual_ret'] > 0 else '-'
    p_sign = '+' if l['pred_ret'] > 0 else '-'
    match  = 'HIT ' if a_sign == p_sign else 'MISS'
    print(f"  {l['date']}  act={a_sign}{abs(l['actual_ret']):.4f}  pred={p_sign}{abs(l['pred_ret']):.5f}  {match}  r2_log={l['r2']:+.3f}")
