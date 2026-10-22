# prior-strength-interp

**Does telling a language model the wrong answer make it less confident in the right one — and can we find where that happens inside the network?**

This project measures how prior strength (sycophantic framing) affects Qwen2.5-1.5B's confidence across layers, using a logit lens and causal activation patching.

<!-- TODO: add "Open in Colab" badge once the notebook is in the repo -->

## What it does

- **100 factual questions**, each with a correct and an incorrect answer
- **Five prompt variants per question:** `neutral`, `mention`, `weak`, `strong`, `expert` (each suggesting the wrong answer with increasing confidence)
- **Logit lens:** project hidden states at every layer through the model's norm + LM head to track P(right answer) across depth
- **Filter** to questions the model answers correctly under neutral framing (63/100)
- **Causal test:** compute the "strong minus neutral" direction in residual stream space, subtract it at layers 22–27 via forward hooks, and compare against a random vector of equal norm

## Results

| Condition | P(right answer) at final layer |
|-----------|-------------------------------:|
| neutral   | 0.175 |
| expert    | 0.126 |
| strong    | 0.078 |
| mention   | 0.074 |
| weak      | 0.048 |

All biased conditions suppressed model confidence in the correct answer. The effect is present even at the logit-lens level across intermediate layers, suggesting the prior influences representations well before the final layer.

## Model

[`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B) · float16 · T4 GPU (Google Colab)

## Limitations

- Small model (1.5B) — results may not generalize to larger models
- 63 questions is a modest sample
- Logit lens is an approximation; intermediate layers weren't trained to be decoded this way
- Causal subtraction is crude — a more principled approach would use DAS or activation patching per component
