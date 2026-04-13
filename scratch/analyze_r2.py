import json, numpy as np

with open('simulation_state.json') as f:
    data = json.load(f)

logs = data['gold']['diagnostic_logs']
print(f'test_idx: {data["gold"]["test_idx"]}  logs: {len(logs)}')
print()

if not logs:
    print('No logs yet')
else:
    actuals = np.array([l['actual_ret'] for l in logs])
    preds   = np.array([l['pred_ret'] for l in logs])
    vols    = np.array([l['vol_mult'] for l in logs])
    cals    = np.array([l.get('calibrated_pred', l['pred_ret'] * l['vol_mult']) for l in logs])

    print('=== SIGN DISTRIBUTION ===')
    print(f'pred_ret > 0: {(preds>0).sum()}/{len(preds)} ({(preds>0).mean():.0%})')
    print(f'pred_ret < 0: {(preds<0).sum()}/{len(preds)} ({(preds<0).mean():.0%})')
    print(f'actual  > 0:  {(actuals>0).sum()}/{len(actuals)}')
    print()
    print('=== MAGNITUDE ===')
    print(f'avg |actual|:   {np.mean(np.abs(actuals)):.5f}  ({np.mean(np.abs(actuals))*100:.3f}%)')
    print(f'avg |pred_raw|: {np.mean(np.abs(preds)):.5f}  ({np.mean(np.abs(preds))*100:.3f}%)')
    print(f'avg |cal_pred|: {np.mean(np.abs(cals)):.5f}  ({np.mean(np.abs(cals))*100:.3f}%)')
    print(f'avg vol_mult:   {np.mean(vols):.3f}')
    print()

    sst = np.sum((actuals - np.mean(actuals))**2)
    r2_raw = 1 - np.sum((actuals - preds)**2)/(sst+1e-9)
    r2_cal = 1 - np.sum((actuals - cals)**2)/(sst+1e-9)
    print('=== R2 ===')
    print(f'R2 raw pred:       {r2_raw:.4f}')
    print(f'R2 calibrated:     {r2_cal:.4f}')
    
    # What R2 WOULD be if we just used direction + mean magnitude
    dir_acc = np.mean(np.sign(preds) == np.sign(actuals))
    print(f'Directional accuracy: {dir_acc:.1%}')

    print()
    print('=== LAST 10 ENTRIES ===')
    for l in logs[-10:]:
        cal = l.get('calibrated_pred', l['pred_ret'] * l['vol_mult'])
        print(f"  {l['date']}  actual={l['actual_ret']:+.4f}  pred_raw={l['pred_ret']:+.5f}  cal={cal:+.4f}  r2={l['r2']:.3f}  flush={l['is_flush']}")
