# Steering Pipeline - Quick Start

**📍 You are here:** Quick Start Guide → Run steering in 6 steps  
**🏠 Main menu:** [README.md](README.md) | **🔧 Customize:** [docs/STEERING_ADVANCED.md](docs/STEERING_ADVANCED.md) | **💡 Theory:** [docs/THEORY.md](docs/THEORY.md) | **❓ Help:** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

Reduce hallucinations in LLM outputs via Modified Inference-Time Intervention (MITI) - applying steering vectors to attention heads during generation.

**For advanced topics:** See [docs/STEERING_ADVANCED.md](docs/STEERING_ADVANCED.md)  
**For troubleshooting:** See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)  
**For theory:** See [docs/THEORY.md](docs/THEORY.md)

## What This Does

Identifies which attention heads contribute to hallucinations, calculates steering vectors for those heads, and applies real-time interventions during generation to reduce hallucination rates.

**How it works:** Generate baseline → Capture activations → Find best heads → Steer during generation → Measure improvement

**Pipeline:** 6 steps (baseline → ensemble → activations → vectors → steering → analysis)

**Dataset requirements:** CSV files (MCQ or open-ended format)

💡 **Need to install?** Complete setup in [README.md](README.md#installation) first

## Model Preset System - Important!

**Only 2 scripts use model presets:**

| Script | Uses Preset? | Flag | Preset Models |
|--------|--------------|------|---------------|
| ✅ `baseline_run.py` | **YES** | `--model {llama\|qwen\|gemma}` | 3 preconfigured models |
| ✅ `grab_activation_ITI_attnhookz.py` | **YES** | `--model {llama\|qwen\|gemma}` | 3 preconfigured models |
| ❌ `baseline_ensemble_voter.py` | **NO** | None | Works with any baseline |
| ❌ `steer_vector_calc_ITI.py` | **NO** | None | Model-agnostic |
| ❌ `steering_experiment.py` | **NO** | None | Uses baseline model automatically |
| ❌ `cross_experiment_analysis.py` | **NO** | None | Model-agnostic |

**Preset models (in `steer/utils/steer_common_utils.py`):**
- `--model llama` → Llama-3-8B-Instruct (1024 heads, 32 layers)
- `--model qwen` → Qwen2.5-7B-Instruct (784 heads, 28 layers)
- `--model gemma` → Gemma-2-9B-IT (672 heads, 42 layers)

**Can I use other models?**
- ✅ **YES** - Any model in the [TransformerLens Model Library](https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html)
- Add your model to `MODEL_CONFIGS` in `steer/utils/steer_common_utils.py` with correct layer count and head configuration
- See [docs/STEERING_ADVANCED.md#adding-new-models](docs/STEERING_ADVANCED.md#adding-new-models) for instructions

**Key insight:** After baseline and activation capture (Steps 1-3), all other steps work with ANY model because they use saved data, not the model directly.

---

## Quick Reference

**⚠️ IMPORTANT:** All commands in this guide are examples. You **must** replace dataset paths with your own CSV files. Look for sections marked "Adapt this example" in each step.

**How many baseline runs?** → **5 times** (reduces API noise via ensemble voting)  
**Which K values?** → Try `10 100 500 1000` for Llama (1%, 10%, 50%, 98% of heads)  
**Which α values?** → Start with `0.5 1.0 2.5 5.0 10.0`, then narrow around best  
**Which datasets?** → 1 primary (target) + 2 secondary (validation like MMLU/HellaSwag)

## 6-Step Workflow

### Step 1: Generate Baselines - Run 5 Times

**Important:** You need to run baseline generation **5 times** with the same parameters to create an ensemble. Each run creates a timestamped directory automatically.

**⚠️ Adapt this example to your dataset:**

```bash
# Example - Replace paths with your actual dataset:
python -m steer.baseline_run \
  --model llama \
  --dataset-path ./data/YOUR_DATASET.csv \
  --dataset-format non_mcq \
  --num-samples 1000 \
  --output-dir ./data/baseline_results/individual/YOUR_DATASET_NAME
```

**Required flags:**
- `--model {llama|qwen|gemma}` - Model preset to use
- `--dataset-path ./data/YOUR_DATASET.csv` - **Replace with your CSV file path**
- `--dataset-format {mcq|non_mcq|mmlu|hellaswag}` - Format matching your data
- `--num-samples N` - Number of samples to process
- `--output-dir ./data/baseline_results/individual/YOUR_NAME` - **Use a descriptive name**

**Optional flags:**
- `--device-id 0` - GPU device to use (default: 0)
- `--batch-size 8` - Batch size (default: 1)
- `--max-tokens 120` - Max tokens to generate (default: 120)

**What happens:**
- Loads model (`--model llama` uses preset Llama-3-8B-Instruct)
- Generates unsteered answers for each question
- Evaluates with GPT-4 (scores: 0=correct, 1=hallucination, 2=API fail)
- Saves results with timestamp

**You'll see:**
```
Loading model: Llama-3-8B-Instruct...
Generating baseline: 100%|██████████| 1000/1000
Evaluating with GPT-4: 100%|██████████| 1000/1000
✓ Saved to BASELINE_YOUR_DATASET_20260216_123456/
```

**Outputs:** 5 `BASELINE_*_timestamp/` directories in your specified output-dir:
- `baseline_evaluation.pkl` - Hallucination scores per sample
- `baseline_texts.pkl` - Generated answers
- `baseline_config.json` - Configuration used

**Why 5 times?** Ensemble voting reduces noise from GPT-4 evaluation variability. Each run gets evaluated independently, then Step 2 uses majority voting for robust labels.

**Dataset formats:**
- `non_mcq` - Open-ended QA (columns: context, question, answer)
- `mcq` - Multiple choice (columns: context, question, choices, answerKey)
- `mmlu`, `hellaswag` - Benchmark formats

💡 **Using Qwen or Gemma?** Change `--model qwen` or `--model gemma`  
💡 **Need custom model?** See [docs/STEERING_ADVANCED.md#adding-new-models](docs/STEERING_ADVANCED.md#adding-new-models)

---

### Step 2: Create Ensemble Baseline

**⚠️ Adapt this example to your paths:**

```bash
# Example - Replace with your actual directories from Step 1:
python -m steer.baseline_ensemble_voter \
  --parent-dir ./data/baseline_results/individual/YOUR_DATASET_NAME \
  --output-dir ./data/baseline_results/ensembled_YOUR_DATASET_NAME
```

**Required flags:**
- `--parent-dir` - Directory containing the 5 baseline runs from Step 1
- `--output-dir` - Output directory for ensemble results

**What happens:** Majority voting across 5 runs → More robust baseline  
**Outputs:** `BASELINE_ENSEMBLE_VOTED_{TIMESTAMP}/` directory with voted labels

**Note:** The script automatically finds all `BASELINE_*` directories in the parent-dir and performs majority voting across them.

---

### Step 3: Capture Activations

**⚠️ Adapt this example to your dataset:**

```bash
# Example - Can use different dataset than baseline:
python -m steer.grab_activation_ITI_attnhookz \
  --model llama \
  --dataset-path ./data/YOUR_ACTIVATION_DATASET.csv \
  --dataset-format mcq \
  --num-samples 1000 \
  --output-dir ./data/ITI/activations
```

**Required flags:**
- `--model {llama|qwen|gemma}` - Must match the model from Step 1
- `--dataset-path ./data/YOUR_DATASET.csv` - **Can be different from Step 1**
- `--dataset-format {mcq|non_mcq|mmlu|hellaswag}` - Format matching your data
- `--num-samples N` - Number of samples to process

**Optional flags:**
- `--device-id 0` - GPU device to use (default: 0)
- `--batch-size 8` - Batch size (default: 8)
- `--output-dir` - Output directory (default: ./data/ITI/activations)

**What happens:**
- Captures attention head outputs (`attn.hook_z`) during generation
- Can use **different dataset** than baseline (tests generalization)
- Evaluates each sample with GPT-4 for labels

**Outputs:** `ITI_ACTIVATIONS_{DATASET_NAME}_{TIMESTAMP}/` with HDF5 files per layer

---

### Step 4: Calculate Steering Vectors

**⚠️ Replace timestamp with actual directory from Step 3:**

```bash
# Example - Find actual directory name in ./data/ITI/activations/
python -m steer.steer_vector_calc_ITI \
  --h5-dir ./data/ITI/activations/ITI_ACTIVATIONS_YOUR_DATASET_20260216_123456 \
  --top-k 10 100 500 1000 \
  --output-dir ./data/ITI/steering_vector/llama/default
```

**Required flags:**
- `--h5-dir` - Path to activations directory from Step 3 (includes timestamp)
- `--top-k` - List of K values to test (number of heads to steer)

**Optional flags:**
- `--output-dir` - Output directory (default: ./data/ITI/steering_vector)

**What happens:** Trains classifiers per head → Ranks by accuracy → Computes steering directions  
**Outputs:** `iti_intervention_config_top{K}.pkl` for each K value (10, 100, 500, 1000)

💡 **Find the actual directory:** Run `ls ./data/ITI/activations/` or check Windows Explorer - look for directories starting with `ITI_ACTIVATIONS_` and use the most recent one with your dataset name and timestamp.

---

### Step 5: Run Steering Experiments

**⚠️ Replace all paths with your actual directories:**

**Test on primary dataset** (where you want to reduce hallucinations):

```bash
# Example - Replace timestamps and paths with your actual directories:
python -m steer.steering_experiment \
  --baseline-dir ./data/baseline_results/ensembled_YOUR_DATASET/BASELINE_ENSEMBLE_VOTED_TIMESTAMP \
  --iti-config-path ./data/ITI/steering_vector/llama/default/iti_intervention_config_top100.pkl \
  --steering-strength 0.5 1.0 2.5 5.0 10.0 \
  --output-dir ./data/ITI/steering_experiment/round1/eval_on_YOUR_DATASET
```

**Required flags:**
- `--baseline-dir` - Ensemble baseline directory from Step 2 (includes timestamp)
- `--iti-config-path` - Steering config file from Step 4 (choose a K value)
- `--steering-strength` - List of alpha values to test
- `--output-dir` - Output directory for results

**Optional flags:**
- `--device-id 0` - GPU device (default: 0)
- `--batch-size 8` - Batch size (default: 8)
- `--model {llama|qwen|gemma}` - Should match baseline (default: llama)

**Repeat for secondary datasets** (MMLU, HellaSwag) to check capability preservation:

```bash
# Example - After generating baselines for MMLU (Steps 1-2):
python -m steer.steering_experiment \
  --baseline-dir ./data/baseline_results/ensembled_mmlu/BASELINE_ENSEMBLE_VOTED_TIMESTAMP \
  --iti-config-path ./data/ITI/steering_vector/llama/default/iti_intervention_config_top100.pkl \
  --steering-strength 0.5 1.0 2.5 5.0 10.0 \
  --output-dir ./data/ITI/steering_experiment/round1/eval_on_mmlu
```

**What happens:** Applies steering at each α strength → Evaluates with GPT-4  
**Outputs:** `STEERING_{K}_{ALPHA}/` subdirectories with results

💡 **Repeat for each K value** from Step 4 (10, 100, 500, 1000) by changing the `--iti-config-path`

---

### Step 6: Find Optimal (K, α) Configuration

**⚠️ Adapt paths to your actual experiment directories:**

```bash
# Example - Replace with your actual experiment directories:
python -m steer.cross_experiment_analysis \
  --primary-dir ./data/ITI/steering_experiment/round1/eval_on_YOUR_DATASET \
  --secondary-dir-1 ./data/ITI/steering_experiment/round1/eval_on_VALIDATION_DATASET_1 \
  --secondary-dir-2 ./data/ITI/steering_experiment/round1/eval_on_VALIDATION_DATASET_2 \
  --output-dir ./data/ITI/cross_analysis/round1
```

**Required flags:**
- `--primary-dir` - Results directory for primary dataset (from Step 5)

**Optional flags:**
- `--secondary-dir-1` - First validation dataset results (optional)
- `--secondary-dir-2` - Second validation dataset results (optional)
- `--output-dir` - Output directory (default: ./data/ITI/cross_analysis)

**What happens:** Finds best (K, α) that:
- Maximizes hallucination reduction on primary dataset
- Preserves performance on secondary datasets (≤1% degradation)

**Outputs:**
- `report.json` - Ranked configurations with metrics
- Heatmaps - Visual performance across parameters

**Check:** Open `report.json` → Look for configs marked `PASS` → Use highest-ranked

---

## Expected Performance

### Hallucination Reduction on NQ-Swap

Steering successfully reduced hallucination rates on NQ-Swap (Longpre et al., 2022) across models:

- **Gemma-2-9B**: Improved by **3.30pp** from a baseline of 35.18%
  - Optimal hyperparameters: K = 7, α = 28.20
  - Lowest hallucination rate makes it ideal for faithfulness-critical use cases
  
- **Llama-3-8B**: Improved by **5.88pp** from a baseline of 58.82%
  - Optimal hyperparameters: K = 15, α = 3.50
  
- **Qwen2.5-7B**: Achieved the largest reduction at **19.25pp** from a baseline of 75.48%
  - Optimal hyperparameters: K = 418, α = 3.25
  - Reduction achieved under the relaxed 3% threshold
  - Cannot achieve faithfulness without significantly affecting general abilities at stricter thresholds

**Key insight:** The optimal hyperparameters varied substantially across models, highlighting the importance of model-specific tuning.

### Impact on General Abilities

Steering affected general abilities differently across models:

**MMLU (Hendrycks et al., 2021) - Domain Knowledge:**
- Llama: -0.88pp
- Gemma: -0.52pp
- Qwen: -1.41pp

**HellaSwag (Zellers et al., 2019) - Commonsense Reasoning:**
- Llama: -0.53pp
- Gemma: **+4.60pp** (improvement!)
- Qwen: -2.66pp

**Analysis:** Faithfulness entangles with different capabilities per model:
- **Gemma**: Faithfulness aligns with commonsense reasoning, so steering for one enhances the other, while opposing domain knowledge
- **Llama**: Domain knowledge opposes faithfulness more strongly than commonsense reasoning
- **Qwen**: Commonsense reasoning more strongly opposed to faithfulness than domain knowledge

**Recommendation:** For faithfulness-critical applications requiring preserved commonsense reasoning, Gemma should be used.

### Steering Behavior Analysis

Faithfulness is concentrated in fewer heads for Gemma versus distributed across many for Llama and Qwen, which may explain Gemma's lower negative impact on general performance.

---

## Understanding Results

**Hallucination scores:**
- **0** = Correct answer
- **1** = Hallucination (incorrect)
- **2** = API failure (excluded from metrics)

**Key metrics:**
- **Baseline rate:** % hallucinations before steering
- **Steered rate:** % hallucinations after steering
- **Absolute reduction:** Percentage point decrease (e.g., 45% → 30% = **15pp** reduction)

**What to look for:**
1. Rate decreases as α increases (up to a point)
2. Sweet spot: α where primary reduction peaks before secondary degradation
3. In `report.json`: Configurations marked `PASS` meet constraints

💡 **Details:** See [docs/STEERING_ADVANCED.md#understanding-results](docs/STEERING_ADVANCED.md#understanding-results)

---

## Common Issues & Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| **CUDA out of memory** | Add `--batch-size 1` to commands |
| **API rate limits** | Scripts retry automatically - wait |
| **Steering has no effect** | Try more heads (higher K) or stronger α |
| **Import errors** | Run from project root: `cd c:/Users/enqiy/dso-internship-all` |
| **"No valid samples"** | Re-run baselines with stable API |

💡 **More issues?** → [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## Complete Minimal Example

**⚠️ IMPORTANT:** This is a complete example showing the workflow. You **must** adapt the dataset paths to your own data.

**Goal:** Reduce hallucinations on a QA dataset with Llama-3-8B

```bash
# 0. Setup
cd c:/Users/enqiy/dso-internship-all
# Create .env file in project root with Azure OpenAI credentials

# 1. Generate baselines - RUN THIS COMMAND 5 TIMES (creates 5 timestamped directories)
python -m steer.baseline_run \
  --model llama \
  --dataset-path ./data/YOUR_DATASET.csv \
  --dataset-format non_mcq \
  --num-samples 200 \
  --output-dir ./data/baseline_results/individual/YOUR_DATASET

# 2. Ensemble - Combines the 5 baseline runs
python -m steer.baseline_ensemble_voter \
  --parent-dir ./data/baseline_results/individual/YOUR_DATASET \
  --output-dir ./data/baseline_results/ensembled_YOUR_DATASET

# 3. Activations - Can use different dataset than baseline
python -m steer.grab_activation_ITI_attnhookz \
  --model llama \
  --dataset-path ./data/YOUR_ACTIVATION_DATASET.csv \
  --dataset-format mcq \
  --num-samples 200 \
  --output-dir ./data/ITI/activations

# 4. Steering vectors - Replace TIMESTAMP with actual from step 3 output
#    (Check ./data/ITI/activations/ for directory name)
python -m steer.steer_vector_calc_ITI \
  --h5-dir ./data/ITI/activations/ITI_ACTIVATIONS_YOUR_DATASET_TIMESTAMP \
  --top-k 10 100 \
  --output-dir ./data/ITI/steering_vector/llama/default

# 5. Steering experiment - Replace TIMESTAMP with actual from step 2
#    (Check ./data/baseline_results/ensembled_YOUR_DATASET/ for directory name)
python -m steer.steering_experiment \
  --baseline-dir ./data/baseline_results/ensembled_YOUR_DATASET/BASELINE_ENSEMBLE_VOTED_TIMESTAMP \
  --iti-config-path ./data/ITI/steering_vector/llama/default/iti_intervention_config_top100.pkl \
  --steering-strength 1.0 2.5 5.0 \
  --output-dir ./data/ITI/steering_experiment/round1/eval_on_YOUR_DATASET

# 6. Check results - K and alpha values appear in directory name
# Open file: ./data/ITI/steering_experiment/round1/eval_on_YOUR_DATASET/STEERING_100_{ALPHA}/ALL_RESULTS_ANALYSIS.json
# Or run cross-experiment analysis if you have secondary datasets
```

**Key points:**
- **Step 1:** Run the exact same command 5 times - each creates a timestamped directory
- **Timestamps:** After Steps 2, 3, check the output to find actual directory names
- **Dataset paths:** Replace `YOUR_DATASET` with your actual CSV filename
- **K=100:** This example uses K=100 (steer top 100 heads), adjust based on your Step 4 results

---

## Dataset Format Examples

### Supported Dataset Types

**1. MCQ Format** (`--dataset-format mcq`):

**Required columns:**
- `context` - Background knowledge/passage
- `question` - Question text
- `choices` - JSON string of answer options: `[{"label":"A","text":"..."}, ...]`
- `answerKey` - Correct option letter (e.g., "A", "B", "C")

**Example:**
```csv
context,question,choices,answerKey
"Paris is capital.","Capital of France?","[{""label"":""A"",""text"":""Paris""},{""label"":""B"",""text"":""London""}]","A"
```

**2. Non-MCQ Format** (`--dataset-format non_mcq`):

**Required columns:**
- `context` - Background knowledge/passage
- `question` - Question text
- `answer` - Ground truth answer text

**Example:**
```csv
context,question,answer
"Paris is capital.","Capital of France?","Paris"
```

**3. MMLU Format** (`--dataset-format mmlu`):
- Standard MMLU benchmark format
- Automatically handled by dataset loader

**4. HellaSwag Format** (`--dataset-format hellaswag`):
- Standard HellaSwag benchmark format
- Automatically handled by dataset loader

### Adding Custom Dataset Types

**Step 1:** Check context size requirements
- Ensure your model can handle: `len(context + question) + MAX_ANSWER_TOKENS`
- Llama-3-8B max: 8192 tokens
- Truncate context if needed to leave room for generation

**Step 2:** Create adapter in `steer/utils/dataset_formats.py`:
```python
def load_custom_format(df):
    """
    Convert your dataset to standard format.
    Must return columns: context, question, answer (or choices + answerKey for MCQ)
    """
    # Your transformation logic
    return df
```

**Step 3:** Register in dataset loader:
```python
# In steer/utils/dataset_formats.py
DATASET_LOADERS = {
    'mcq': load_mcq,
    'non_mcq': load_non_mcq,
    'mmlu': load_mmlu,
    'hellaswag': load_hellaswag,
    'custom': load_custom_format  # Add your loader
}
```

**Step 4:** Use with `--dataset-format custom` flag

⚠️ **Important:** Always verify your dataset fits within model context limits before running the pipeline.

---

## Which Files to Modify

**To change model (only affects Steps 1 & 3):**
- Edit `steer/utils/steer_common_utils.py` → Add to `MODEL_CONFIGS`
- Or use existing presets: `--model llama`, `--model qwen`, `--model gemma`

**To change dataset format:**
- Edit `steer/utils/dataset_formats.py` → Add custom adapter

**To change parameters:**
- K values: `steer_vector_calc_ITI.py` → `--top-k` flag
- α values: `steering_experiment.py` → `--steering-strength` flag

💡 **Complete guide:** [docs/STEERING_ADVANCED.md](docs/STEERING_ADVANCED.md)

---

## Next Steps

**Customize the pipeline:**
- [Add new models](docs/STEERING_ADVANCED.md#adding-new-models)
- [Create custom dataset formats](docs/STEERING_ADVANCED.md#custom-dataset-formats)
- [Understand parameter selection](docs/STEERING_ADVANCED.md#understanding-steering-parameters)
- [Optimize performance](docs/STEERING_ADVANCED.md#performance-optimization)

**Get help:**
- [Fix common errors](docs/TROUBLESHOOTING.md)
- [Understand MITI theory](docs/THEORY.md)
- [Complete CLI reference](docs/API_REFERENCE.md#steering-pipeline-commands)
