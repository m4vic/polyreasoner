import json
import glob
import os

results_dir = os.path.dirname(os.path.abspath(__file__))
# Check for vision files
files = sorted(glob.glob(os.path.join(results_dir, "results", "vision", "exp1_*.json")))

print(f"{'Model':<25} | {'Run':<4} | {'Acc':>5} | {'Iters':>5} | {'BestIt':>6} | {'Sunk':>4} | {'Time':>6} | Stop Reason")
print("-" * 120)

models_data = {}

for f in files:
    try:
        with open(f, 'r') as file:
            data = json.load(file)
            
            model = data.get('model', 'Unknown')
            # remove prefix
            for prefix in ['ollama/', 'openai/', 'anthropic/']:
                model = model.replace(prefix, '')
                
            run_tag = os.path.basename(f).split('_')[3] # exp1_model_dataset_run_...
            
            acc = data.get('best_accuracy', 0.0)
            if acc is None or acc == float('inf') or acc == float('-inf'):
                acc = 0.0
            iters = data.get('total_iterations', 0) or 0
            best_iter = data.get('best_iteration', 0) or 0
            sunk = data.get('sunk_cost_episodes', 0) or 0
            time_s = data.get('total_time_seconds', 0.0) or 0.0
            reason = data.get('stop_reason', 'Unknown')
            
            if reason and len(reason) > 50:
                reason = reason[:47] + "..."
                
            print(f"{model:<25} | {run_tag:<4} | {acc:.3f} | {iters:5} | {best_iter:6} | {sunk:4} | {time_s:6.1f} | {reason}")
            
            if model not in models_data:
                models_data[model] = []
            models_data[model].append(acc)
            
    except Exception as e:
        print(f"Error reading {f}: {e}")

print("\n--- AVERAGES ---")
for m, accs in sorted(models_data.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True):
    avg = sum(accs) / len(accs)
    print(f"{m:<25} : {avg:.3f} (over {len(accs)} runs)")
