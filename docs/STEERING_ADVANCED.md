# Steering Pipeline - Advanced Guide

**📍 You are here:** Advanced Guide → Customize steering  
**🏠 Main menu:** [README.md](../README.md) | **⚡ Quick Start:** [README_Steering.md](../README_Steering.md) | **💡 Theory:** [THEORY.md](THEORY.md) | **❓ Help:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

Advanced configuration, customization, and deep-dive topics for the MITI steering pipeline.

**Haven't run steering yet?** Start with [README_Steering.md](../README_Steering.md)

## Table of Contents

- [Adding New Models](#adding-new-models)
- [Custom Dataset Formats](#custom-dataset-formats)
- [Understanding Steering Parameters](#understanding-steering-parameters)
- [Activation Capture Details](#activation-capture-details)
- [Steering Vector Calculation](#steering-vector-calculation)
- [Advanced Analysis Techniques](#advanced-analysis-techniques)
- [Performance Optimization](#performance-optimization)
- [API Cost Management](#api-cost-management)
- [Reproducibility Settings](#reproducibility-settings)
- [Code Architecture](#code-architecture)

## Adding New Models

### Model Requirements

For a model to work with the steering pipeline:

1. **TransformerLens support** - Model must be loadable via `HookedTransformer.from_pretrained()`
2. **Attention structure** - Must have standard `attn.hook_z` (attention head outputs)
3. **Chat template** - For instruction-tuned models, must have a recognized format

### Step-by-Step Integration

**1. Check TransformerLens compatibility**

```python
from transformer_lens import HookedTransformer

try:
    model = HookedTransformer.from_pretrained("your-model-id")
    print(f"✓ Model loaded successfully")
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  Attention heads: {model.cfg.n_heads}")
    print(f"  Total heads: {model.cfg.n_layers * model.cfg.n_heads}")
    print(f"  Has attn.hook_z: {'attn.hook_z' in model.hook_dict}")
except Exception as e:
    print(f"✗ Model not compatible: {e}")
```

**2. Add model configuration to `steer/utils/steer_common_utils.py`**

Find the `MODEL_CONFIGS` dictionary and add your model:

```python
MODEL_CONFIGS = {
    # Existing models...
    "llama": {...},
    "qwen": {...},
    "gemma": {...},
    
    # Your new model
    "your_model": {
        "name": "your-model-id",
        "heads": 1024,  # n_layers * n_heads
        "template": "llama",  # or "gemma", "qwen", "custom"
    }
}
```

**3. Define chat template if custom**

If your model uses a unique chat format, add a formatter function:

```python
def format_your_model_prompt(context, question, choices=None):
    """Format prompt for your custom model."""
    if choices:  # MCQ format
        choices_text = "\n".join([
            f"{c['label']}. {c['text']}" for c in choices
        ])
        return f"""<|system|>You are a helpful assistant.<|end|>
<|user|>{context}

{question}

{choices_text}

Answer with only the letter.<|end|>
<|assistant|>"""
    else:  # Non-MCQ format
        return f"""<|system|>You are a helpful assistant.<|end|>
<|user|>{context}

{question}<|end|>
<|assistant|>"""

# Register in MODEL_CONFIGS
MODEL_CONFIGS["your_model"]["template"] = "custom"
MODEL_CONFIGS["your_model"]["formatter"] = format_your_model_prompt
```

**4. Test model with baseline run**

```bash
python -m steer.baseline_run \
  --model your_model \
  --dataset-path ./data/test.csv \
  --dataset-format mcq \
  --num-samples 10 \
  --output-dir ./data/test_baseline
```

Check:
- ✓ Model loads without errors
- ✓ Generations look reasonable
- ✓ GPT-4 can evaluate outputs
- ✓ No shape mismatches in activations

**5. Calculate optimal K values**

```python
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
total_heads = n_layers * n_heads

# Suggested K values (1%, 10%, 50%, 98%)
k_values = [
    int(total_heads * 0.01),
    int(total_heads * 0.10),
    int(total_heads * 0.50),
    int(total_heads * 0.98)
]
print(f"Recommended K values: {k_values}")
```

### Example: Adding Mistral-7B

```python
# In steer/utils/steer_common_utils.py
MODEL_CONFIGS["mistral"] = {
    "name": "mistralai/Mistral-7B-Instruct-v0.2",
    "heads": 256,  # 32 layers × 8 heads
    "template": "mistral"
}

def format_mistral_prompt(context, question, choices=None):
    if choices:
        choices_text = "\n".join([f"{c['label']}. {c['text']}" for c in choices])
        return f"[INST] {context}\n\n{question}\n\n{choices_text}\n\nAnswer with only the letter. [/INST]"
    else:
        return f"[INST] {context}\n\n{question} [/INST]"

MODEL_CONFIGS["mistral"]["formatter"] = format_mistral_prompt
```

Then use:
```bash
python -m steer.baseline_run --model mistral ...
```

## Custom Dataset Formats

### Supported Formats Overview

Four built-in formats in `steer/utils/dataset_formats.py`:

1. **mcq** - General multiple choice questions
2. **non_mcq** - Open-ended QA
3. **mmlu** - MMLU benchmark format
4. **hellaswag** - HellaSwag benchmark format

### Adding a New Dataset Format

**Step 1: Define loader function**

Edit `steer/utils/dataset_formats.py`:

```python
def load_your_format(dataset_path: str) -> pd.DataFrame:
    """
    Load your custom dataset format.
    
    Returns:
        DataFrame with columns: context, question, choices (if MCQ), answerKey (if MCQ), answer (if non-MCQ)
    """
    df = pd.read_csv(dataset_path)
    
    # Transform to standard format
    # Example: Rename columns
    df = df.rename(columns={
        'passage': 'context',
        'query': 'question',
        'options': 'choices',
        'correct': 'answerKey'
    })
    
    # Example: Transform choices format
    def parse_choices(choices_str):
        # Your parsing logic
        # Return list of dicts: [{"label": "A", "text": "..."}, ...]
        pass
    
    df['choices'] = df['choices'].apply(parse_choices)
    
    return df

# Register format
DATASET_LOADERS['your_format'] = load_your_format
```

**Step 2: Use in scripts**

```bash
python -m steer.baseline_run \
  --dataset-format your_format \
  --dataset-path ./data/your_dataset.csv
```

### Dataset Quality Guidelines

**Size:**
- Minimum: 100 samples (testing only)
- Recommended: 1000-2000 samples
- Larger datasets → better steering vectors but higher API cost

**Balance:**
- Ideal: 30-70% hallucination rate
- Too easy (>90% correct): Hard to find useful heads
- Too hard (<20% correct): Model may need fine-tuning, not steering

**Quality checks:**
```python
import pandas as pd

df = pd.read_csv('your_dataset.csv')

# Check required columns
required = ['context', 'question']
assert all(c in df.columns for c in required), "Missing required columns"

# Check for nulls
print(f"Null counts:\n{df.isnull().sum()}")

# Check context lengths
df['context_len'] = df['context'].str.len()
print(f"Context length stats:\n{df['context_len'].describe()}")

# Recommend: Most contexts 200-2000 characters
```

## Understanding Steering Parameters

### K (Number of Heads)

**What it controls:** How many top-ranked attention heads to intervene on

**Considerations:**
- **Low K (1-5%):** Surgical, minimal side effects, may be insufficient
- **Medium K (10-20%):** Balanced, often most effective
- **High K (50%+):** Aggressive, risk of capability degradation

**Finding optimal K:**
1. Start with fixed α (e.g., 2.0)
2. Test multiple K values: `10 100 500 1000` (adjust for model size)
3. Check cross-analysis report for best K at that α
4. Narrow K range around best, test with varied α

**Model-specific K ranges:**

```python
# Llama-3-8B (1024 heads)
K_TEST = [10, 50, 100, 200, 500, 1000]

# Qwen2.5-7B (784 heads)  
K_TEST = [8, 40, 78, 156, 392, 768]

# Gemma-2-9B (672 heads)
K_TEST = [7, 35, 67, 134, 336, 659]
```

### α (Alpha - Steering Strength)

**What it controls:** Magnitude of intervention applied to head activations

**Mathematical effect:**
```python
# During generation at each layer:
steered_activation = original_activation + α * steering_vector
```

**Considerations:**
- **α < 1:** Subtle nudge, may not affect outputs
- **α = 1-5:** Typical effective range
- **α > 10:** Strong intervention, risk of nonsense outputs

**Multi-round strategy:**

**Round 1: Wide search**
```bash
--steering-strength 0.5 1.0 2.0 5.0 10.0 20.0
```

**Round 2: Narrow around best (example: best was α=2.0)**
```bash
--steering-strength 1.5 1.75 2.0 2.25 2.5
```

**Round 3: Fine-tune (example: best was α=2.25)**
```bash
--steering-strength 2.1 2.15 2.2 2.25 2.3 2.35 2.4
```

### Threshold Parameters in Analysis

```bash
python -m steer.cross_experiment_analysis \
  --primary-dir ... \
  --secondary-dir-1 ... \
  --secondary-dir-2 ... \
  --secondary-threshold 1.0 \
  --secondary-threshold-2 1.0
```

**secondary-threshold:** Maximum allowed hallucination rate **increase** (percentage points) on secondary-dir-1 (typically MMLU)

**secondary-threshold-2:** Maximum allowed increase on secondary-dir-2 (typically HellaSwag)

**Example:**
- Baseline MMLU: 15% hallucination rate
- Threshold: 1.0
- After steering: 15.8% → PASS (0.8pp increase ≤ 1.0)
- After steering: 16.5% → FAIL (1.5pp increase > 1.0)

**Tuning thresholds:**
- **Strict (0.5):** Minimal degradation allowed, fewer configs pass
- **Balanced (1.0):** Typical choice, allows minor degradation
- **Lenient (2.0):** More configs pass, check outputs manually

## Activation Capture Details

### What Gets Captured

**Hook:** `attn.hook_z` (attention head outputs)

**Shape:** `[batch_size, seq_len, n_heads, d_head]`

**Extraction:**
- For each sample, capture activations at **last generated token position**
- Reshape to `[n_heads, d_head]` (one vector per head)
- Label with hallucination score (0 correct, 1 hallucination)

**Storage:** HDF5 files per layer

```python
# File: activations_layer_15.h5
# Datasets:
#   - activations: shape [n_samples, n_heads * d_head]
#   - labels: shape [n_samples] (0 or 1)
#   - head_indices: shape [n_heads] (which heads in which layers)
```

### Why Last Token Position?

**Reasoning:**
- Accumulates information from entire generation
- Model's "confidence state" at end of answer
- Most correlated with final output quality

**Alternative positions tested in detection pipeline:**
- BOS token: Too early, lacks generation context
- Last prompt token: Misses generation behavior
- First generated: Insufficient context
- **Last generated: Best empirical results** ⭐

### Batch Size Considerations

```bash
--batch-size 12  # Typical for 16GB VRAM
```

**Memory usage:**
- Model weights: ~15GB (for 7-9B models)
- Activations per sample: ~50-200MB (depends on seq length)
- Batch processing: batch_size × activation_size

**If OOM:**
```bash
--batch-size 1  # Minimum, slower but safe
```

**Optimal:**
```bash
--batch-size 16  # Faster if GPU allows
```

## Steering Vector Calculation

### Classification Per Head

For each attention head across all layers:

1. **Extract features:** Head's d_head-dimensional activations
2. **Split data:** 80% train, 20% test (stratified)
3. **Train classifier:** Logistic regression to predict 0 (correct) vs 1 (hallucination)
4. **Evaluate:** Compute accuracy on test set
5. **Rank:** Sort heads by test accuracy

**Why logistic regression?**
- Simple, interpretable
- Fast to train (critical for 500-1000 heads)
- Sufficient for linear separability of hallucination signal

### Steering Direction Extraction

For top-K heads:

```python
# Pseudocode
mass_mean_diffs = []

for head in top_k_heads:
    activations_correct = head_activations[labels == 0]  # Label 0
    activations_halluc = head_activations[labels == 1]   # Label 1
    
    mean_correct = activations_correct.mean(axis=0)
    mean_halluc = activations_halluc.mean(axis=0)
    
    # Direction: from hallucination toward correct
    direction = mean_correct - mean_halluc
    
    mass_mean_diffs.append(direction)

# Store for each head with layer, head_idx metadata
```

**Interpretation:**
- Steering vector points from "hallucination space" toward "correct space"
- During inference, push activations in this direction
- α scales magnitude of push

### Configuration Files

**Output:** `iti_intervention_config_top{K}.pkl`

```python
# Structure
{
    'heads_details': [
        {
            'layer': 15,
            'head_idx': 3,
            'accuracy': 0.87,
            'mass_mean_diff': array([...]),  # Steering vector
        },
        # ... top K heads
    ],
    'metadata': {
        'total_heads_ranked': 1024,
        'k': 100,
        'dataset': 'faitheval',
        'n_samples': 1000
    }
}
```

### Validation Across Datasets

**Strategy:**
1. **Primary dataset:** Where you want to reduce hallucinations (e.g., NQ-Swap)
2. **Secondary dataset 1:** General knowledge test (e.g., MMLU)
3. **Secondary dataset 2:** Commonsense reasoning (e.g., HellaSwag)

**Why multiple secondary datasets?**
- MMLU: Tests factual knowledge preservation
- HellaSwag: Tests reasoning and pragmatics
- Different capabilities → more robust validation

**Red flags:**
- Primary improves but both secondaries degrade → overfitting
- One secondary OK, other degrades → capability tradeoff
- All degrade → α too high or K poorly chosen

### Iterative Refinement

**Round 1: Exploration**
- Wide K range: `10 100 500 1000`
- Wide α range: `0.5 1.0 2.5 5.0 10.0`
- Goal: Identify promising regions

**Round 2: Exploitation**
- Narrow K: Around best from Round 1 ± factor of 2-3
- Narrow α: Around best ± 50%
- Goal: Fine-tune configuration

**Round 3: Validation**
- Fix K from Round 2
- Very narrow α: Best ± 10%
- Test on held-out dataset
- Goal: Verify generalization


## Reproducibility Settings

### Random Seeds

**Where randomness occurs:**
1. **Model generation:** Temperature, top-p sampling
2. **Train/test split:** In classifier training
3. **API variance:** GPT-4 evaluation (non-deterministic)

**Control methods:**

**1. Model generation (line 123-125 in baseline_run.py):**
```python
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
```

**2. Classifier training (in steer_vector_calc_ITI.py):**
```python
train_test_split(..., random_state=42)
LogisticRegression(..., random_state=42)
```

**3. API variance:**
```bash
# Ensemble voting across 5 runs reduces variance
# Variance typically ±2-3% across runs
```

### Deterministic Results

**To maximize reproducibility:**

**1. Fix all seeds**
```python
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
```

**2. Disable CUDA nondeterminism**
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**3. Use temperature=0 for generation**
```python
# In generate call
outputs = model.generate(..., temperature=0.0, do_sample=False)
```

**4. Run baselines 5+ times and ensemble**
```bash
# Voting across multiple runs reduces API variance
```

**Limitation:** GPT-4 evaluation is inherently non-deterministic even with temperature=0. Ensemble voting is the mitigation.

## Code Architecture

### Pipeline Flow

```
User Input
    ↓
[baseline_run.py] × 5 runs
    ├─ Load model
    ├─ Generate answers
    ├─ Evaluate with GPT-4
    └─ Save: baseline_evaluation.pkl, baseline_texts.pkl
    ↓
[baseline_ensemble_voter.py]
    ├─ Load 5+ individual baseline runs
    ├─ Majority vote per sample
    └─ Save: ensembled baseline_evaluation.pkl
    ↓
[grab_activation_ITI_attnhookz.py]
    ├─ Load model with TransformerLens
    ├─ Register attn.hook_z hooks
    ├─ Generate with activation capture
    ├─ Extract last token activations
    ├─ Evaluate with GPT-4 (for labels)
    └─ Save: activations_layer_*.h5
    ↓
[steer_vector_calc_ITI.py]
    ├─ Load all activation HDF5 files
    ├─ For each head:
    │   ├─ Train logistic regression
    │   ├─ Evaluate accuracy
    │   └─ Store results
    ├─ Rank heads by accuracy
    ├─ Calculate mass mean diffs (steering directions)
    └─ Save: iti_intervention_config_top{K}.pkl
    ↓
[steering_experiment.py] (repeat for each K, α)
    ├─ Load baseline ensemble
    ├─ Load steering config
    ├─ Load model
    ├─ For each α:
    │   ├─ Apply steering during generation
    │   ├─ Evaluate with GPT-4
    │   └─ Save results
    └─ Save: ALL_RESULTS_ANALYSIS.json
    ↓
[cross_experiment_analysis.py]
    ├─ Load primary results
    ├─ Load secondary results
    ├─ Calculate metrics per (K, α)
    ├─ Apply constraints
    ├─ Rank configurations
    ├─ Generate heatmaps
    └─ Save: report.json, heatmaps
```

### Key Modules

**steer/utils/steer_common_utils.py**
- MODEL_CONFIGS: Model metadata
- Prompt formatting functions
- Shared utility functions

**steer/utils/dataset_formats.py**
- DATASET_LOADERS: Format parsers
- load_mcq(), load_non_mcq(), etc.

**steer/utils/eval_model_steer.py**
- Azure GPT-4 evaluation
- Retry logic, caching
- evaluate_answer_gptoai()

**steer/utils/steer_files_utils.py**
- File I/O helpers
- Pickle save/load
- Directory traversal

**steer/utils/steer_graphs_utils.py**
- Heatmap generation
- Plotting functions

### Extension Points

**Add new model:**
- Edit: `steer/utils/steer_common_utils.py` → MODEL_CONFIGS

**Add new dataset format:**
- Edit: `steer/utils/dataset_formats.py` → DATASET_LOADERS

**Add new evaluation method:**
- Edit: `steer/utils/eval_model_steer.py` → evaluate_answer_*()

**Add new metric:**
- Edit: `cross_experiment_analysis.py` → metric calculation section

**Add new hook type:**
- Edit: `grab_activation_ITI_attnhookz.py` → hook registration and extraction

---

**See also:**
- [Quick Start Guide](../README_STEERING.md)
- [Theory: How MITI Works](THEORY.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [API Reference](API_REFERENCE.md)
- [Main README](../README.md)
