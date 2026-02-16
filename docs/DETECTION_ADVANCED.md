# Detection Pipeline - Advanced Guide

**📍 You are here:** Advanced Guide → Customize detection  
**🏠 Main menu:** [README.md](../README.md) | **⚡ Quick Start:** [README_DETECTION.md](../README_DETECTION.md) | **❓ Help:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

Advanced configuration, customization, and architecture details for the hallucination detection pipeline.

**Haven't run detection yet?** Start with [README_DETECTION.md](../README_DETECTION.md)

## Table of Contents

- [Understanding config.py](#understanding-configpy)
- [Hook System Explained](#hook-system-explained)
- [Layer Selection Strategy](#layer-selection-strategy)
- [Token Position Schemes](#token-position-schemes)
- [Custom Scoring Configuration](#custom-scoring-configuration)
- [Class Imbalance Handling](#class-imbalance-handling)
- [Extending the Pipeline](#extending-the-pipeline)
- [Pipeline Architecture](#pipeline-architecture)
- [Output File Specifications](#output-file-specifications)
- [Command-Line Override Reference](#command-line-override-reference)
- [Performance Optimization](#performance-optimization)

## Understanding config.py

### Model Configuration

```python
# Lines 26-28
MODEL_NAME = "llama3_8b_instruct"
HUGGINGFACE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
TRANSFORMER_LENS_MODEL_NAME = HUGGINGFACE_MODEL_ID  # Auto-set, don't change
```

**To change models:**
1. Visit [TransformerLens Model Table](https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html)
2. Find your model and copy `name.default.alias` value
3. Update `HUGGINGFACE_MODEL_ID` in config.py
4. Set `MODEL_NAME` to a descriptive identifier (used in file naming)

**Supported model families:**
- GPT-2, GPT-Neo, GPT-J
- Llama, Llama-2, Llama-3
- Mistral, Mixtral
- Qwen, Qwen2
- Gemma, Gemma-2

### Dataset Configuration

```python
# Lines 32-34
QA_DATASETS = [
    {'path': './data/squad_clean.csv', 'samples': 50000}
]
```

**Multiple dataset support:**
```python
QA_DATASETS = [
    {'path': './data/squad_clean.csv', 'samples': 40000},
    {'path': './data/trivia_qa.csv', 'samples': 10000}
]
```

**Required CSV columns:**
- `knowledge` - Context passage
- `question` - Question text
- `answer` - Ground truth answer

**Dataset tips:**
- Place in `./data/` directory
- 10k-100k samples recommended for robust classifiers
- Balance question difficulty and domain coverage

### Experiment Settings

```python
# Lines 48-60
EXPERIMENT_ID = "squad"           # Unique identifier (used in paths)
BATCH_SIZE = 8                    # Samples per batch (affects GPU memory)
DEVICE_ID = 0                     # GPU device ID
TOTAL_SAMPLES = 50000             # Total samples to process
NUM_CHUNKS = 4                    # Chunks for distributed processing
CHUNK_ID = 0                      # Which chunk to process (0 to NUM_CHUNKS-1)
```

**Chunking strategy:**
- Each chunk processes `TOTAL_SAMPLES / NUM_CHUNKS` samples
- Run chunks in parallel on different GPUs for speed
- Outputs saved in separate `chunk_{ID}` directories
- Classifier automatically merges all chunks

**Batch size tuning:**
- Larger = faster but more GPU memory
- Start with 8, reduce if OOM errors
- Typical range: 1-32 depending on model size

### Activation Capture Settings

```python
# Lines 62-63
START_LAYER = 8                   # First layer (inclusive)
END_LAYER = 18                    # Last layer (inclusive)
SKIP_LAYERS = {}                  # Layers to exclude (set of ints)
```

**Layer selection guidelines:**
- **Early layers (0-N/3):** Low-level features, less semantic
- **Middle layers (N/3-2N/3):** Best for hallucination detection ⭐
- **Late layers (2N/3-N):** High-level features, task-specific

**Model-specific recommendations:**
- **Llama-3-8B (32 layers):** 8-18 or 10-20
- **GPT-2 (12 layers):** 4-10
- **Llama-2-7B (32 layers):** 8-18

```python
# Lines 65-85 - Hook configuration
ADDITIONAL_HOOKS = [
    "attn.hook_z",          # Attention head outputs ⭐
    "mlp.hook_pre",         # MLP inputs ⭐
    # Add more hooks as needed
]

TOKEN_SCHEMES = [
    "last_generated"        # Last token position ⭐
]
```

**Minimal recommended config:**
```python
ADDITIONAL_HOOKS = ["attn.hook_z"]
TOKEN_SCHEMES = ["last_generated"]
START_LAYER = 10
END_LAYER = 20
```

This captures only attention outputs at the last token for layers 10-20, balancing performance and speed.

## Hook System Explained

### What Are Hooks?

Hooks are named locations in the model where you can capture internal activations during forward pass. TransformerLens provides standardized hook names across different model architectures.

### Available Hooks by Category

#### Residual Stream (3 hooks)

```python
"hook_resid_pre"    # Input to transformer layer (before attention)
"hook_resid_mid"    # After attention, before MLP
"hook_resid_post"   # After MLP (layer output)
```

**Use case:** Tracking information flow through the layer  
**Shape:** `[batch, seq_len, d_model]`

#### Attention (9 hooks)

```python
"attn.hook_z"              # Attention head outputs ⭐ RECOMMENDED
"attn.hook_pattern"        # Attention weights (where model looks)
"attn.hook_attn_scores"    # Raw attention scores (pre-softmax)
"attn.hook_q"              # Query vectors
"attn.hook_k"              # Key vectors
"attn.hook_v"              # Value vectors
"attn.hook_rot_q"          # Rotary position embeddings for queries
"attn.hook_rot_k"          # Rotary position embeddings for keys
"hook_attn_out"            # Combined attention output
```

**Best for hallucination detection:**
- `attn.hook_z` - Captures per-head outputs, strongest signal
- `attn.hook_pattern` - Shows attention distribution

**Shape of attn.hook_z:** `[batch, seq_len, n_heads, d_head]`

#### MLP (4 hooks)

```python
"mlp.hook_pre"         # MLP input (before activation) ⭐ RECOMMENDED
"mlp.hook_post"        # MLP output (after activation)
"mlp.hook_pre_linear"  # Before final linear layer
```

**Use case:** Captures feed-forward transformations  
**Shape:** `[batch, seq_len, d_model]` or `[batch, seq_len, d_mlp]`

#### LayerNorm (2 hooks)

```python
"ln1.hook_normalized"  # Pre-attention normalization
"ln2.hook_normalized"  # Pre-MLP normalization
```

**Use case:** Normalized representations  
**Shape:** `[batch, seq_len, d_model]`

### Hook Recommendations by Use Case

**Minimal (fastest):**
```python
ADDITIONAL_HOOKS = ["attn.hook_z"]
```

**Balanced (recommended):**
```python
ADDITIONAL_HOOKS = [
    "attn.hook_z",
    "mlp.hook_pre"
]
```

**Comprehensive (research):**
```python
ADDITIONAL_HOOKS = [
    "hook_resid_pre",
    "attn.hook_z",
    "attn.hook_pattern",
    "mlp.hook_pre",
    "mlp.hook_post",
    "hook_resid_post"
]
```

### How Activations Are Captured

1. **Generator runs:** Model processes input, generates answer
2. **Hook triggers:** At specified points, activation tensors are captured
3. **Token selection:** Based on TOKEN_SCHEMES, specific positions are extracted
4. **Storage:** Saved to HDF5 files per layer/hook/token combination
5. **Classifier loads:** Activations become features for ML classifiers

## Layer Selection Strategy

### Why Middle Layers Work Best

**Research findings:**
- Early layers: Syntactic patterns, less semantic meaning
- Middle layers: Abstract concepts, factuality signals ← Best for hallucination
- Late layers: Task-specific adaptations, compressed for output

### How to Find Optimal Layers

**Method 1: Binary search**
1. Run classifier on all layers (START_LAYER=0, END_LAYER=N-1)
2. Check `comprehensive_analysis_report.html` for layer performance
3. Identify peak performance range
4. Re-run with narrowed range for faster processing

**Method 2: Heuristic**
- For N-layer model, start with layers floor(N/3) to floor(2N/3)
- Example: 32-layer model → layers 10-21

**Method 3: Literature**
- Check papers on your specific model
- Llama models: layers 8-20 consistently work well
- GPT-2: layers 4-10

### Practical Examples

```python
# GPT-2 (12 layers) - Quick test
START_LAYER = 4
END_LAYER = 10

# Llama-3-8B (32 layers) - Comprehensive
START_LAYER = 8
END_LAYER = 24

# Llama-3-8B (32 layers) - Fast focused
START_LAYER = 12
END_LAYER = 18

# Custom: Skip specific problematic layers
START_LAYER = 10
END_LAYER = 20
SKIP_LAYERS = {13, 17}  # Exclude layers 13 and 17
```

## Token Position Schemes

### Available Schemes

```python
TOKEN_SCHEMES = [
    "bos_token",           # Position 0 (beginning of sequence)
    "last_prompt_token",   # Last token of input prompt
    "first_generated",     # First generated token
    "last_generated"       # Last generated token ⭐
]
```

### When to Use Each Scheme

**bos_token:**
- Captures global context encoding
- Less useful for generation-specific hallucinations
- Good for input understanding analysis

**last_prompt_token:**
- Marks transition from prompt to generation
- Useful for prompt-dependent hallucinations
- Captures final prompt state

**first_generated:**
- Initial generation decision point
- May predict trajectory of full generation
- Lightweight (only 1 token captured)

**last_generated:** ⭐ **RECOMMENDED**
- Captures accumulated generation state
- Best correlation with hallucination labels
- Reflects confidence at end of generation
- Most reliable in practice

### Multiple Token Schemes

You can capture multiple positions simultaneously:

```python
TOKEN_SCHEMES = [
    "last_prompt_token",  # Prompt understanding
    "last_generated"      # Final generation state
]
```

The classifier will train separate models for each and select the best.

### Token Position Details

**What "last_generated" actually means:**
- After model generates full answer
- Truncated at first period if `FIRST_PERIOD_TRUNCATION = True`
- Otherwise, at max token limit (`MAX_ANSWER_TOKENS`)
- Activation captured from that specific sequence position

## Custom Scoring Configuration

### The Custom Scoring System

The classifier uses a custom metric to select the best configuration, balancing multiple objectives:

```python
# Lines 315-321 in config.py
CUSTOM_SCORING = {
    'beta': 2.5,              # F-beta weight for hallucination recall
    'w': 0.8,                 # Blend weight for hallucination class
    'gamma': 1.2,             # MCC gate power (strictness)
    'blend': 'geom'           # Blend method: 'arith' or 'geom'
}
```

### Parameters Explained

**beta** (default: 2.5)
- Controls F-beta score for class 1 (hallucinations)
- Higher values emphasize recall over precision
- beta=1 → balanced F1, beta=2.5 → 2.5× weight on recall
- **Use case:** Catching all hallucinations is more important than false alarms

**w** (default: 0.8)
- Weight for class 1 in weighted blend with class 0
- Range: 0.0 to 1.0
- 0.8 means 80% weight on hallucination detection, 20% on correct detection
- **Use case:** Prioritize minority class (hallucinations)

**gamma** (default: 1.2)
- Powers the MCC (Matthews Correlation Coefficient) gating function
- Higher values make gate more strict (penalize low MCC more)
- Acts as quality threshold
- **Use case:** Ensure overall balanced performance

**blend** (default: 'geom')
- `'arith'` → Arithmetic mean: (`w` * `score_1` + (1-`w`) * `score_0`)
- `'geom'` → Geometric mean: (`score_1`^`w` * `score_0`^(1-`w`))
- Geometric mean more sensitive to low scores
- **Use case:** 'geom' prevents one very bad metric

### How Custom Scoring Works

```python
# Pseudocode
f_beta_class_1 = (1 + beta^2) * precision * recall / (beta^2 * precision + recall)
f_beta_class_0 = similar for class 0

if blend == 'geom':
    blend_fb = f_beta_1^w * f_beta_0^(1-w)
else:
    blend_fb = w * f_beta_1 + (1-w) * f_beta_0

mcc_gate = mcc^gamma

custom_score = blend_fb * mcc_gate
```

**Best classifier:** Highest custom_score across all layer/hook/token combinations

### Tuning Recommendations

**Conservative (prefer precision):**
```python
CUSTOM_SCORING = {
    'beta': 1.0,      # Balanced F1
    'w': 0.5,         # Equal weight
    'gamma': 1.5,     # Strict MCC gate
    'blend': 'geom'
}
```

**Aggressive (prefer recall):**
```python
CUSTOM_SCORING = {
    'beta': 3.0,      # Heavy recall emphasis
    'w': 0.9,         # Focus on hallucinations
    'gamma': 1.0,     # Lenient MCC gate
    'blend': 'arith'
}
```

**Balanced (default):**
```python
CUSTOM_SCORING = {
    'beta': 2.5,
    'w': 0.8,
    'gamma': 1.2,
    'blend': 'geom'
}
```

## Class Imbalance Handling

### Auto-Detection

```python
# Lines 269-270 in config.py
HANDLE_IMBALANCE = True
IMBALANCE_THRESHOLD = 0.4
```

**How it works:**
1. Classifier checks label distribution
2. If minority class < 40%, imbalance detected
3. Applies class weights: `class_weight='balanced'` in sklearn
4. Uses stratified splits to maintain label ratios

### When Imbalance Matters

**Scenarios:**
- High-accuracy models (>90% correct) → Few hallucinations
- Filtered datasets with mostly correct answers
- Small sample sizes amplify imbalance

**Effects without handling:**
- Classifier predicts majority class for everything
- High accuracy but useless (e.g., 95% accuracy by always predicting "correct")
- Zero recall for hallucinations

**With handling:**
- Balances loss function across classes
- Enforces equal importance in training
- Better minority class recall

### Manual Configuration

```python
# Disable auto-detection
HANDLE_IMBALANCE = False

# Or adjust threshold
IMBALANCE_THRESHOLD = 0.3  # More sensitive (triggers at 30%)
```

## Extending the Pipeline

### Custom Dataset Formats

**Step 1:** Ensure CSV has required columns:
- `knowledge`
- `question`
- `answer`

**Step 2:** Add custom loader in `helpers/custom_loader.py`:
```python
def load_custom_dataset(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        'passage': 'knowledge',
        'query': 'question',
        'ground_truth': 'answer'
    })
    return df
```

**Step 3:** Import in generator.py and use

### Custom Evaluation Metrics

**Step 1:** Edit `helpers/eval_model.py`:
```python
def custom_evaluator(knowledge, question, answer, model_answer):
    # Your evaluation logic
    # Return: 0 (correct) or 1 (hallucination)
    pass
```

**Step 2:** Add environment variable for mode selection
```python
EVAL_METHOD = os.getenv('EVAL_METHOD', 'azure')  # 'azure' or 'custom'
```

## Pipeline Architecture

### Data Flow

```
1. Generator (core/generator.py)
   ├─ Load datasets from QA_DATASETS
   ├─ For each chunk:
   │  ├─ Process batch of samples
   │  ├─ Generate answer with model
   │  ├─ Capture activations at hooks
   │  ├─ Evaluate correctness (0/1)
   │  └─ Save to HDF5 + CSV
   └─ Output: ./data/activations/{EXPERIMENT_ID}/chunk_{ID}/

2. Classifier (core/classifier.py)
   ├─ Load all activation HDF5 files
   ├─ For each layer/hook/token combination:
   │  ├─ Train logistic regression classifier
   │  ├─ Cross-validate performance
   │  ├─ Compute custom score
   │  └─ Save metrics
   ├─ Select best configuration
   ├─ Generate comprehensive HTML report
   └─ Output: ./data/classifier/classifier_run_{TIMESTAMP}/

3. Evaluator (core/evaluate.py)
   ├─ Load best trained classifier
   ├─ Load evaluation dataset
   ├─ For each sample:
   │  ├─ Generate answer with model
   │  ├─ Capture activations (same config as training)
   │  ├─ Predict hallucination probability
   │  └─ Evaluate correctness
   ├─ Generate evaluation report
   └─ Output: .../evaluation_run_{TIMESTAMP}/

4. Real-Time Inference (core/real_time_inference.py)
   ├─ Load classifier + metadata
   ├─ Load model (auto-detected from metadata)
   ├─ Interactive loop:
   │  ├─ User enters question
   │  ├─ Generate answer
   │  ├─ Capture activations
   │  ├─ Predict hallucination probability
   │  └─ Display to user
```

### Key Modules

**helpers/activation_utils.py**
- Manages TransformerLens hook registration
- Captures and stores activations during forward pass
- Handles different hook types and shapes

**helpers/model_manager.py**
- Centralizes model loading
- Caches loaded models to avoid redundant loading
- Handles device placement (CPU/GPU)

**helpers/token_manager.py**
- Tokenization wrapper
- Manages token position tracking
- Handles truncation and padding

**helpers/checkpoint.py**
- Saves intermediate results
- Enables resume after interruption
- Atomic file operations for safety

**helpers/eval_model.py**
- Azure GPT-4 evaluation
- Retry logic for API failures
- Caches responses to avoid redundant API calls

**helpers/custom_scoring.py**
- Implements custom metric calculation
- Handles imbalance-aware scoring
- Computes F-beta, MCC, blended scores

## Output File Specifications

### Generator Outputs

**Location:** `./data/activations/{EXPERIMENT_ID}/chunk_{CHUNK_ID}/`

**Files:**
1. **Activation HDF5 files** - `activations_L{layer}_{hook}_{token_scheme}.h5`
   - Datasets: `activations` (shape: [n_samples, activation_dim]), `labels` (shape: [n_samples])
   - One file per layer/hook/token combination

2. **Results CSV** - `results_chunk_{CHUNK_ID}.csv`
   - Columns: `question`, `knowledge`, `answer`, `model_answer`, `correct`, `sample_id`
   - One row per sample processed

3. **Config metadata** - `config_metadata.json`
   ```json
   {
     "model_name": "llama3_8b_instruct",
     "huggingface_model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
     "start_layer": 8,
     "end_layer": 18,
     "hooks": ["attn.hook_z", "mlp.hook_pre"],
     "token_schemes": ["last_generated"],
     "batch_size": 8,
     "experiment_id": "squad"
   }
   ```

4. **Logs** - `pipeline_full_*.log`, `pipeline_errors_*.log`

### Classifier Outputs

**Location:** `./data/classifier/classifier_run_{TIMESTAMP}/`

**Files:**
1. **Best classifier** - `best_classifier_model.joblib`
   - Trained sklearn LogisticRegression model
   - Ready for joblib.load() and .predict()

2. **Results CSV** - `results_all_groups.csv`
   - All layer/hook/token combinations tested
   - Columns: layer, hook, token_scheme, accuracy, precision, recall, f1, roc_auc, custom_score

3. **HTML report** - `comprehensive_analysis_report.html`
   - Open in browser for interactive exploration
   - Performance metrics, confusion matrices, feature importance

4. **Run metadata** - `run_metadata.json`
   - Configuration snapshot, timestamp, best config

### Evaluator Outputs

**Location:** `./data/classifier/classifier_run_{RUN_ID}/evaluation_run_{TIMESTAMP}/`

**Files:**
1. **Predictions** - `evaluation_results.csv`
   - Per-sample predictions with probabilities
   - Columns: sample_id, question, answer, model_answer, true_label, predicted_label, hallucination_probability

2. **Metrics** - `evaluation_metrics.json`
   ```json
   {
     "accuracy": 0.85,
     "precision": 0.82,
     "recall": 0.78,
     "f1": 0.80,
     "roc_auc": 0.88,
     "confusion_matrix": [[800, 50], [120, 380]]
   }
   ```

3. **HTML report** - `evaluation_report.html`

## Command-Line Override Reference

### Generator Overrides

```bash
python -m core.generator \
  --chunk-id 0 \
  --device-id 0 \
  --model-name "llama3_8b_instruct" \
  --huggingface-model-id "meta-llama/Meta-Llama-3-8B-Instruct" \
  --batch-size 16 \
  --first-period-truncation True
```

### Classifier Overrides

```bash
python -m core.classifier \
  --experiment-id squad \
  --gpu-ids 0,1,2 \
  --num-gpus 3 \
  --max-batches 10 \
  --max-batch-files 5
```

### Evaluator Overrides

```bash
python -m core.evaluate \
  --experiment-id squad \
  --dataset-csv-path ./data/custom.csv \
  --batch-size 32 \
  --sample-size 10000 \
  --device-id 1 \
  --first-period-truncation False
```

**Pro tip:** Command-line args override config.py without modifying the file, useful for experimentation.

## Performance Optimization

### Speed Improvements

**1. Reduce activation capture:**
```python
# Minimal hooks
ADDITIONAL_HOOKS = ["attn.hook_z"]
TOKEN_SCHEMES = ["last_generated"]

# Narrow layer range
START_LAYER = 12
END_LAYER = 18
```

**2. Increase batch size:**
```python
BATCH_SIZE = 32  # If GPU memory allows
```

**3. Parallelize chunks:**
```bash
# Terminal 1
python -m core.generator --chunk-id 0 --device-id 0 &
# Terminal 2
python -m core.generator --chunk-id 1 --device-id 1 &
# Terminal 3
python -m core.generator --chunk-id 2 --device-id 2 &
```

### Memory Optimization

**1. Reduce batch size:**
```python
BATCH_SIZE = 2  # Minimum for persistent efficiency
```

**2. Use gradient checkpointing (requires code modification):**
```python
model.cfg.use_split_qkv_input = True
```

**3. Enable mixed precision (requires code modification):**
```python
import torch
with torch.autocast(device_type='cuda', dtype=torch.float16):
    # Model forward pass
```

### Storage Optimization

**1. Compress HDF5 files:**
```python
# In helpers/activation_utils.py, modify h5 creation:
f.create_dataset('activations', data=acts, compression='gzip', compression_opts=9)
```

**2. Reduce sample count:**
```python
TOTAL_SAMPLES = 10000  # Instead of 50000 for testing
```

**3. Clean up intermediate files:**
```bash
# After classifier training, can delete activation HDF5s (keep CSVs)
rm ./data/activations/${EXPERIMENT_ID}/chunk_*/activations_*.h5
```

---

**See also:**
- [Quick Start Guide](../README_DETECTION.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [Main README](../README.md)
