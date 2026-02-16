# FAQ & Troubleshooting

**📍 You are here:** Troubleshooting → Fix problems  
**🏠 Main menu:** [../README.md](../README.md) | **⚡ Quick Starts:** [Detection](../README_DETECTION.md) · [Steering](../README_Steering.md) | **💡 Theory:** [THEORY.md](THEORY.md)

---

**For implementation instructions:** See Quick Start guides ([Detection](../README_DETECTION.md) | [Steering](../README_Steering.md))  
**For theoretical background:** See [THEORY.md](THEORY.md)

## Table of Contents

- [General Questions](#general-questions)
- [Technical Questions](#technical-questions)
- [Practical Questions](#practical-questions)
- [Reproducibility & Determinism](#reproducibility--determinism)
- [Common Issues & Solutions](#common-issues--solutions)
- [Troubleshooting Reference](#troubleshooting-reference)

## General Questions

### Can I use this pipeline without Azure OpenAI? I have OpenAI API keys.

**Yes, but requires code modification:**

1. Edit `helpers/eval_model.py` and `steer/utils/eval_model_steer.py`
2. Replace `openai.AzureOpenAI` with `openai.OpenAI`
3. Update authentication to use `api_key=` instead of Azure-specific parameters
4. Change `.env` variables accordingly

**Alternative approach:**
Use any evaluation method and manually create `baseline_evaluation.pkl` files with scores 0/1/2.

### Can I run this without GPT-4 evaluation at all?
No.

### Why ensemble voting instead of averaging scores?

**Reason:** Scores are categorical (0, 1, 2), not continuous.

- Averaging would create meaningless decimals (what does score 0.7 mean?)
- Majority voting preserves discrete labels
- More robust to outliers (1 wrong evaluation in 5 runs won't dominate)

**Example:**
- Sample scores across 5 runs: [0, 0, 1, 0, 0]
- Majority vote: 4 votes for 0 → Label as 0 (faithful)
- Average: 0.2 → What does this mean? Unclear!

### Can I use this for other tasks besides hallucination reduction?

**Yes! The pipeline is generalizable:**

1. **Replace hallucination labels** with any binary property:
   - Toxic/Safe language
   - Formal/Informal tone
   - Biased/Unbiased statements
   - On-topic/Off-topic responses

2. **Capture activations** with task-specific labels

3. **Steering will push** toward labeled behavior

**Note:** Original paper (Li et al.) demonstrated this for truthfulness. The technique is general.

### How do I know if my steering vectors are working?

**Check these signs:**

✅ **Good signs:**
1. Hallucination rate decreases with increasing α (at least initially)
2. Top heads show clear separation in activation directions
3. Results degrade with very high α (sign you're pushing too hard - expected)
4. Different K values show different optimal α (expected)

❌ **Bad signs:**
1. No effect at any α → Try different K or check activation quality
2. Hallucination rate increases → May have swapped labels (0↔1)
3. All activations look identical → Dataset too homogeneous

## Technical Questions

**Could you use other hooks?**
Yes! Experiment suggestions:
- `hook_q`, `hook_k`, `hook_v`: Query/key/value projections
- `hook_attn`: Attention patterns
- `hook_resid_pre/post`: Residual stream before/after layer

**What you need to change:**
1. In `grab_activation_ITI_attnhookz.py` line 307: Change `"attn.hook_z"` to your desired hook (e.g., `"attn.hook_q"`)
2. In `steer_vector_calc_ITI.py` line 1091: Change `self.hook = "attn.hook_z"` to match

**What happens automatically:**
- Hook name gets saved to ITI config file by `steer_vector_calc_ITI.py`
- `steering_experiment.py` loads hook name from config automatically
- Hook registration happens automatically using `get_hook_name_for_layer()` function
- No manual changes needed in `steering_experiment.py`

### What if my dataset has more than 2 answer options (MCQ with 5+ choices)?

**Fully supported!** Current MCQ formats handle any number of choices.

**Format correctly:**
```json
[
  {"label":"A","text":"Option 1"},
  {"label":"B","text":"Option 2"},
  {"label":"C","text":"Option 3"},
  {"label":"D","text":"Option 4"},
  {"label":"E","text":"Option 5"}
]
```

**No code changes needed** - the parser handles variable-length choice arrays.

### Can I use multiple GPUs for a single script?

**Scripts are designed for single-GPU execution.**

**For multi-GPU parallelization:**
- ✅ Run multiple experiments in parallel with different `--device-id`
- ❌ Don't run same experiment on multiple GPUs simultaneously (will cause conflicts)

**Example:**
```bash
# Terminal 1 - GPU 0
python -m steer.steering_experiment --device-id 0 --iti-config-path config_top10.pkl &

# Terminal 2 - GPU 1
python -m steer.steering_experiment --device-id 1 --iti-config-path config_top100.pkl &
```

## Practical Questions

### Can I pause and resume long-running scripts?

**Some scripts support resumption, but they are not thoroughly tested. Please do not trust it:**

| Script | Resumable? | How |
|--------|-----------|-----|
| `baseline_run.py` | ❌ No | But fast (2-4 hours), just restart |
| `grab_activation_ITI_attnhookz.py` | ✅ Yes | Checkpoints every N batches |
| `steering_experiment.py` | ❌ No | Restart, but loads existing baseline |
| `steer_vector_calc_ITI.py` | ❌ No | But fast (<30 min) |
| All others | ✅ Fast enough | <10 minutes |

**To resume activation capture:**
Just re-run with same `--output-dir`. Script will:
1. Check for existing `.h5` files
2. Skip already-processed layers
3. Continue from last checkpoint

**Example:**
```bash
# Run 1 (interrupted at layer 15)
python -m steer.grab_activation_ITI_attnhookz \
  --output-dir ./data/ITI/activations
# ^C (interrupted)

# Run 2 (resumes from layer 16)
python -m steer.grab_activation_ITI_attnhookz \
  --output-dir ./data/ITI/activations
# Automatically detects existing files and continues
```

## Reproducibility & Determinism

### Random seed control

**All scripts set:**
```python
np.random.seed(42)
torch.manual_seed(42)
```

**But reproducibility is not perfect due to:**

### Sources of Non-Determinism

#### 1. GPT-4 Evaluation
**Nature:** Azure API returns vary even with `temperature=0`

**Impact:**
- ±2-3% hallucination rate across runs
- Most common source of variability

**Solution:**
- Use ensemble voting (5+ baseline runs)
- Focus on trends rather than exact numbers

#### 2. GPU Operations
**Nature:** CUDA kernels may have non-deterministic behavior

**Impact:**
- Minor activation differences across runs (<0.1%)
- Generally negligible impact on final results

**Solution:**
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Enables CUDA deterministic mode (slower but reproducible)
```

#### 3. Parallel Processing
**Nature:** `ProcessPoolExecutor` in vector calculation

**Impact:**
- Deterministic if same number of workers used
- Head rankings should be identical

**Solution:**
```python
# In steer_vector_calc_ITI.py, specify workers:
executor = ProcessPoolExecutor(max_workers=8)  # Fixed number
```

### Ensuring Maximum Reproducibility

```bash
# Set environment variables
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Use consistent device IDs
--device-id 0  # Always same GPU

# Document everything
# Scripts automatically save configuration JSONs
```

### What Should Reproduce Exactly?

✅ **Should be identical:**
- Activation values (given same model weights and inputs)
- Classifier training results (given same activations)
- Head rankings (given same classifier scores)
- Steering vectors (given same head selection)

⚠️ **Will vary slightly:**
- GPT-4 evaluation scores (±2-3%)
- Timing measurements
- Intermediate API failures

### Reporting Results

**When reporting results, include:**
1. Random seed used (default: 42)
2. Number of baseline runs for ensemble (recommend: 5)
3. GPT-4 API version and deployment name
4. Model versions (HuggingFace commit hash)
5. Sample size and dataset version

**Example:**
```
Results: Llama-3-8B on NQ-Swap (1000 samples)
- Random seed: 42
- Baseline runs: 5 (ensemble voting)
- API: Azure GPT-4 (deployment: gpt-4o, version: 2024-12-01)
- Model: meta-llama/Meta-Llama-3-8B-Instruct @ sha256:abc123
- Reduction: 8.3pp ± 0.4pp (95% CI from 3 full pipeline runs)
```

## Common Issues & Solutions

### Baseline Generation Issues

#### API rate limits
```
Error: Rate limit exceeded for GPT-4
```

**Solution:**
- Script automatically retries with exponential backoff
- If persistent: Reduce `--max-workers` (default is 10, try 5 or 1)
- Check Azure API quota limits in portal
- Verify `AZURE_OPENAI_API_VERSION` is correct
- Try different API version: `2024-02-15-preview`

#### Missing API credentials
```
Error: ValueError: Azure OpenAI credentials not found in environment
```

**Solution:**
1. Create `.env` file in project root
2. Add required variables:
   ```
   AZURE_OPENAI_API_KEY=...
   AZURE_OPENAI_ENDPOINT=...
   AZURE_OPENAI_API_VERSION=...
   EVAL_MODEL=...
   ```
3. Verify `.env` is not in `.gitignore` if using git
4. Restart Python session to reload environment

#### Invalid API credentials
```
Error: AuthenticationError: Invalid API key
```

**Solution:**
- Verify `AZURE_OPENAI_API_KEY` from Azure Portal → Keys and Endpoint
- Check `AZURE_OPENAI_ENDPOINT` format: `https://your-resource.openai.azure.com/`
- Ensure `EVAL_MODEL` matches your deployment name exactly (case-sensitive)
- Try regenerating API key in Azure portal

#### API connection timeout
```
Error: Connection timeout or network error
```

**Solution:**
- Script retries automatically (3 attempts with backoff)
- If persistent: Check internet connection
- Verify Azure endpoint is accessible (firewall/VPN issues)
- Try increasing timeout in `eval_model.py` (default: 120s)
- Check Azure service status page

#### Memory errors during baseline
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce `--batch-size` (try 4, 2, or 1)
- Clear CUDA cache: `torch.cuda.empty_cache()`
- Kill other GPU processes: `nvidia-smi` then `kill -9 <PID>`
- Use smaller model if possible

#### Inconsistent evaluations
```
Problem: Same sample gets different scores across runs
```

**Solution:**
- This is **normal** GPT-4 variability!
- Expected variance: ±2-3%
- **Why we use ensemble voting** (runs 5+ times)
- Don't worry unless variance >5%

### Activation Capture Issues

#### TransformerLens import error
```
ModuleNotFoundError: No module named 'transformer_lens'
```

**Solution:**
- Install the custom fork from `requirements.txt`:
  ```bash
  pip install git+https://github.com/j4y-EQ/TransformerLens-Masking-Fixed.git
  ```
- This fork fixes padding/batch size issues that affect the pipeline
- Standard TransformerLens won't work correctly

#### HDF5 file corruption
```
OSError: Unable to open file (file signature not found)
```

**Solution:**
- Scripts use checkpointing
- Delete corrupted `.h5` file
- Re-run with same `--output-dir`
- Script will recreate that layer only

#### Model not found in TransformerLens
```
ValueError: Model alias 'custom-model' not recognized by TransformerLens
```

**Solution:**
- Check [TransformerLens compatibility table](https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html)
- Update `MODEL_CONFIGS` in `steer_common_utils.py`
- Verify model alias is correct
- Some models not supported by TransformerLens

#### Activation capture too slow
```
Problem: 1 sample per minute, will take days
```

**Solution:**
- Increase `--batch-size` (try 16, 24)
- Use `--start-layer` and `--end-layer` to limit layers
- Check GPU utilization: `nvidia-smi`
- May be bottlenecked by:
  - Slow disk I/O (use SSD)
  - API calls (evaluation, not capture)
  - CPU preprocessing (upgrade CPU)

### Steering Calculation Issues

#### No improvement from steering
```
Problem: Steered results identical to baseline
```

**Solution:**
- Try different K values (may have selected wrong heads)
- Try higher α (5.0, 10.0, 20.0)
- Check activation quality:
  - Are faithful/unfaithful activations different?
  - Plot distributions to verify signal
- Check labels: May be swapped (0↔1)

#### All heads have same score
```
Problem: Classifier scores: [0.52, 0.51, 0.53, 0.52, ...]
```

**Solution:**
- Check activation preprocessing - this was a common issue in early pipeline development
- Weak signal between faithful/unfaithful activations
- Dataset may be too homogeneous
- Try:
  - Verify activation capture is working correctly (check `.h5` files)
  - Different dataset with clearer hallucinations
  - More diverse samples
  - Different layer range

### Cross-Analysis Issues

#### No valid samples found
```
Error: No valid samples - all have API failures across experiments
```

**Solution:**
- Check baseline quality (too many API failures)
- May need to re-run baselines with better API stability
- Reduce API load during baseline runs
- Check API quotas and rate limits

#### Directory structure mismatch
```
Error: No STEERING_* directories found in primary-dir
```

**Solution:**
- Verify `--primary-dir` path is correct
- Ensure steering experiments completed successfully
- Check directory names match pattern: `STEERING_*`
- List directory contents to debug

#### Conflicting metrics
```
Problem: Config improves primary but fails secondary
```

**Solution:**
- This is **working as intended**!
- Shows trade-off between tasks
- Options:
  - Increase `--secondary-threshold` to accept larger trade-off
  - Try different K or α values
  - Accept that not all configs pass both constraints

## Troubleshooting Reference

**Quick lookup table for common errors:**

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| `CUDA out of memory` | Batch size too large | Reduce `--batch-size` to 4, 2, or 1 |
| `API rate limit exceeded` | Too many GPT-4 calls | Script retries automatically, wait |
| `TransformerLens model not found` | Model alias incorrect | Check TransformerLens table, update `MODEL_CONFIGS` |
| `No STEERING_* directories found` | Wrong path or incomplete experiments | Verify path, ensure experiments finished |
| `All samples have API failures` | Unstable API during baseline | Re-run baselines with better stability |
| `No valid samples for analysis` | High API failure rate across experiments | Re-run some experiments, check consistency |
| `Steering has no effect` | Wrong K or α too small | Try different K values or higher α (5.0, 10.0) |
| `File corruption` | Interrupted save operation | Delete corrupted file, re-run (checkpoints exist) |
| `Import error: transformer_lens` | Standard package installed | Uninstall, reinstall custom fork from requirements |
| `Negative hallucination reduction` | Wrong steering direction | Check activation labels, may be swapped (0↔1) |
| `Activations are all identical` | Model not generating diverse outputs | Check prompts, use more varied dataset |
| `Cross-analysis finds 0 valid samples` | High API failure rate | Re-run baselines/steering with stable API |
| `Results not reproducible` | Random seed not set | Scripts auto-set seeds, check GPU determinism |
| `Script hangs indefinitely` | API timeout or deadlock | Kill process, restart, reduce batch size |
| `Permission denied on .env` | File permissions wrong | `chmod 600 .env` on Linux/Mac |
| `Module not found` | Missing installation | `pip install -r requirements.txt` |

---

**For implementation details, see [README.md](README.md)**  
**For theoretical background, see [THEORY.md](THEORY.md)**  
**For technical specifications, see [API_REFERENCE.md](API_REFERENCE.md)**
