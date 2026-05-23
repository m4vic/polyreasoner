import json
import glob
import os

results_dir = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(results_dir, "results", "text", "exp2_*.json")))

print(f"{'Reviewer':<22} | {'Coder':<22} | {'Run':<4} | {'Acc':>5} | {'Iters':>5} | {'BestIt':>6} | {'Sunk':>4} | {'Time':>6} | Stop Reason")
print("-" * 130)

pairs_data = {}

for f in files:
    try:
        with open(f, 'r') as file:
            data = json.load(file)
            
            rev = data.get('reviewer_model', 'Unknown')
            cod = data.get('coder_model', 'Unknown')
            
            for prefix in ['ollama/', 'openai/', 'anthropic/']:
                rev = rev.replace(prefix, '')
                cod = cod.replace(prefix, '')
                
            run_tag = "run?"
            parts = os.path.basename(f).split('_')
            for p in parts:
                if p.startswith("run"):
                    run_tag = p
                    break

            acc = data.get('best_accuracy', 0.0)
            if acc is None or acc == float('inf') or acc == float('-inf'):
                acc = 0.0
            iters = data.get('total_iterations', 0) or 0
            best_iter = data.get('best_iteration', 0) or 0
            sunk = data.get('sunk_cost_episodes', 0) or 0
            time_s = data.get('total_time_seconds', 0.0) or 0.0
            reason = data.get('stop_reason', 'Unknown')
            
            if reason and len(reason) > 40:
                reason = reason[:37] + "..."
                
            print(f"{rev:<22} | {cod:<22} | {run_tag:<4} | {acc:.3f} | {iters:5} | {best_iter:6} | {sunk:4} | {time_s:6.1f} | {reason}")
            
            pair = f"{rev} -> {cod}"
            if pair not in pairs_data:
                pairs_data[pair] = []
            pairs_data[pair].append(acc)
            
    except Exception as e:
        print(f"Error reading {f}: {e}")

print("\n--- AVERAGES ---")
for p, accs in sorted(pairs_data.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True):
    avg = sum(accs) / len(accs)
    best = max(accs)
    print(f"{p:<45} : avg={avg:.3f}  best={best:.3f}  (over {len(accs)} runs)")
