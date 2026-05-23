# Same-Model Ablation Controls for Vision and Text Modalities
# This tests whether the dual-agent gain is simply from having a second step,
# or if it truly relies on *cognitive diversity* (different weight priors).

Write-Host "Starting Vision Ablation (qwen2.5-coder:7b -> qwen2.5-coder:7b)"
python runner_critic.py --reviewer-model qwen2.5-coder:7b --coder-model qwen2.5-coder:7b --dataset vision --run-tag ablation1
python runner_critic.py --reviewer-model qwen2.5-coder:7b --coder-model qwen2.5-coder:7b --dataset vision --run-tag ablation2
python runner_critic.py --reviewer-model qwen2.5-coder:7b --coder-model qwen2.5-coder:7b --dataset vision --run-tag ablation3

Write-Host "Starting Text Ablation (qwen2.5-coder:7b -> qwen2.5-coder:7b)"
python runner_critic.py --reviewer-model qwen2.5-coder:7b --coder-model qwen2.5-coder:7b --dataset text --run-tag ablation1
python runner_critic.py --reviewer-model qwen2.5-coder:7b --coder-model qwen2.5-coder:7b --dataset text --run-tag ablation2
python runner_critic.py --reviewer-model qwen2.5-coder:7b --coder-model qwen2.5-coder:7b --dataset text --run-tag ablation3

Write-Host "Ablation runs completed."
