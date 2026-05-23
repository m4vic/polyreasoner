# Thread D — Frontier Baseline Setup

## API Keys Required

You only need the keys for providers you want to use.
The script **skips providers with missing keys automatically**.

### Recommended starting point (all free):
1. **Groq** — fastest, completely free, Llama-3.3-70B + Qwen-QwQ-32B
   - Sign up: https://console.groq.com
   - API key → Dashboard → API Keys

2. **OpenRouter** — aggregator, many free models including Llama-3.1-405B:free
   - Sign up: https://openrouter.ai
   - API key → Settings → Keys

3. **SambaNova** — free credits, Llama-3.1-405B (fastest 405B inference)
   - Sign up: https://cloud.sambanova.ai

4. **Qwen / Alibaba Dashscope** — free tier, Qwen2.5-72B + Qwen-Max
   - Sign up: https://dashscope.aliyuncs.com

5. **OpenAI** — paid, GPT-4o (you already have this key)

## Set Environment Variables

```powershell
# Run in your terminal before executing the benchmark
$env:GROQ_API_KEY       = "gsk_YOUR_KEY_HERE"
$env:OPENROUTER_API_KEY = "sk-or-YOUR_KEY_HERE"
$env:SAMBANOVA_API_KEY  = "YOUR_KEY_HERE"
$env:QWEN_API_KEY       = "sk-YOUR_KEY_HERE"
$env:OPENAI_API_KEY     = "sk-YOUR_KEY_HERE"

# Then run:
python thread_d_frontier_benchmark.py
```

## Cost Estimate
| Provider | Models | Cost for 12 puzzles |
|----------|--------|-------------------|
| Groq | Llama-3.3-70B, Qwen-QwQ-32B | **$0.00** |
| OpenRouter (free models) | Qwen3-30B, Llama-405B | **$0.00** |
| SambaNova | Llama-3.1-405B | **$0.00** (free credits) |
| Qwen Dashscope | Qwen2.5-72B | **~$0.01** (generous free tier) |
| OpenAI | GPT-4o-mini | **~$0.02** |
| OpenAI | GPT-4o | **~$0.15** |
| OpenAI | GPT-4.1 | **~$0.20** |

**Total with all providers: ~$0.38 max**

## Install Dependencies
```powershell
pip install openai ollama
```
