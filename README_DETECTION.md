# Detection Pipeline - Quick Start

**📍 You are here:** Quick Start Guide → Run detection in 3 steps  
**🏠 Main menu:** [README.md](README.md) | **🔧 Customize:** [docs/DETECTION_ADVANCED.md](docs/DETECTION_ADVANCED.md) | **❓ Help:** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

Detect hallucinations in LLM outputs by analyzing internal model activations during text generation.

**For advanced configuration:** See [docs/DETECTION_ADVANCED.md](docs/DETECTION_ADVANCED.md)  
**For troubleshooting:** See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## What This Does

Captures internal activations (attention heads, MLPs, residual stream) during QA generation, trains classifiers to detect hallucination patterns, and evaluates performance on new datasets.

**How it works:** Model generates answers → Activations captured → GPT-4 evaluates correctness → Classifier learns patterns → Test on new data

**Pipeline:** Generator → Classifier → Evaluator

**Key insight - Systematic Exploration:**
1. **Generator phase**: Captures activations at multiple **token positions** (first, last, BOS, EOS) across different **hooks** (attention, MLP, residual) at different **layers** (0 to N)
2. **Classifier phase**: Trains a separate classifier for each **(token position, layer, hook)** combination - this tests which signal best predicts hallucinations
3. **Selection**: Evaluates all combinations and saves the **best-performing classifier** (highest F1/custom score)
4. **Usage**: The best classifier is then used for evaluation on new datasets and interactive testing

**Example:** If Layer 11's attention output (`hook_z`) at the last generated token has the strongest hallucination signal, that specific combination becomes your detector.

**Dataset requirements:** CSV with columns: `knowledge`, `question`, `answer`

💡 **Need to install?** Complete setup in [README.md](README.md#installation) first

## 3-Step Setup

### Step 1: Configure

**Edit `config.py`** - Change these settings:

```python
# 1. Name your experiment (used in output directories)
EXPERIMENT_ID = "my_test"

# 2. Set your dataset
QA_DATASETS = [
    {'path': './data/my_data.csv', 'samples': 1000}
]

# 3. Choose your model (any HuggingFace model)
HUGGINGFACE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# 4. GPU and parallel processing
DEVICE_ID = 0              # GPU to use (0, 1, 2, etc.)
CHUNK_ID = 0               # Which chunk to process (0 to NUM_CHUNKS-1)
NUM_CHUNKS = 1             # Split processing into chunks for parallelization
BATCH_SIZE = 8             # Batch size (reduce if out of memory)

# 5. Samples and buffering
TOTAL_SAMPLES = 1000       # Total samples across all datasets
CHUNK_SIZE = 1000          # Samples per chunk (auto-calculated: TOTAL_SAMPLES / NUM_CHUNKS)
RESULTS_BUFFER_SIZE = 200  # Save results every N samples
```

**Dataset requirements:** 
- CSV file in `./data/` directory
- Must have columns: `knowledge`, `question`, `answer`

**Model requirements:**
- Any model from HuggingFace supported by TransformerLens
- Check compatibility: [TransformerLens Models](https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html)

💡 **Want to change models, layers, or hooks?** → See [docs/DETECTION_ADVANCED.md](docs/DETECTION_ADVANCED.md#quick-modification-guide)

### Step 2: Prepare Dataset

Place your CSV in `./data/` directory with 3 required columns:

**Example:** `./data/squad_clean.csv`
```csv
knowledge,question,answer
"Paris is the capital of France.","What is the capital of France?","Paris"
"The Sun is a star.","What is the Sun?","A star"
```

### Step 3: Check GPU

```bash
# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If you have multiple GPUs, set `DEVICE_ID` in `config.py` or use `--device-id 0` flag.

## Run Commands

**IMPORTANT:** All commands must be run from project root directory.

```bash
cd c:/Users/enqiy/dso-internship-all
```

### Step 1: Generate Activations

```bash
python -m core.generator --chunk-id 0 --device-id 0
```

**Required flags:**
- `--chunk-id 0` - Which chunk to process (0 to NUM_CHUNKS-1)
- `--device-id 0` - GPU device to use (0, 1, 2, etc.)

**Optional flags:**
- `--model-name` - Override model name from config.py
- `--huggingface-model-id` - Override HuggingFace model ID
- `--first-period-truncation true/false` - Truncate at first period

**What happens:**
- Loads model and dataset
- Generates answers for each question
- Captures internal activations at specified layers/hooks
- Saves results every 200 samples

**You'll see:**
```
Loading model meta-llama/Meta-Llama-3-8B-Instruct...
Processing samples: 100%|██████████| 1000/1000
✓ Saved activations to ./data/activations/my_test/chunk_0/
```

**Outputs:** `./data/activations/{EXPERIMENT_ID}/chunk_0/`
- `activations_hook_resid_post_layer_*.h5` - Activation files per layer
- `results_chunk_0.csv` - Generated answers with correctness labels
- `config_metadata.json` - Model and configuration info
- `pipeline_full_*.log` - Detailed logs

**Multi-GPU parallel processing:**
```bash
# Terminal 1 - Process chunk 0 on GPU 0
python -m core.generator --chunk-id 0 --device-id 0

# Terminal 2 - Process chunk 1 on GPU 1
python -m core.generator --chunk-id 1 --device-id 1
```

💡 **Taking too long? Out of memory?** → See [Troubleshooting](#common-issues--quick-fixes)

---

### Step 2: Train Classifiers

```bash
python -m core.classifier --experiment-id my_test --gpu-ids 0 --num-gpus 1
```

**Required flags:**
- `--experiment-id my_test` - Must match EXPERIMENT_ID from Step 1
- `--gpu-ids 0` - GPU IDs to use (comma-separated for multiple)
- `--num-gpus 1` - Number of GPUs (must match gpu-ids count)

**What happens:**
- Loads all captured activations
- Trains logistic regression classifier for each **(layer, hook, token position)** combination
- Tests multiple configurations in parallel (e.g., 32 layers × 3 hooks × 4 positions = 384 classifiers)
- Selects best performer based on custom scoring
- Saves only the best classifier for downstream use

**You'll see:**
```
Loading activations from ./data/activations/my_test/...
Training classifiers: 100%|██████████| 45/45 groups
Best: Layer 11, hook_z, last_generated (F1: 0.87)
✓ Saved to ./data/classifier/classifier_run_20260216_123456/
```

**Outputs:** `./data/classifier/classifier_run_{TIMESTAMP}/`
- `best_classifier_classifier_run_{TIMESTAMP}.pkl` - **Best trained classifier** (use this for evaluation)
- `comprehensive_analysis_report.html` - **Open in browser** to view results
- `results_all_groups.csv` - Performance of all layer/hook/token combinations

**Multi-GPU training:**
```bash
python -m core.classifier --experiment-id my_test --gpu-ids 0,1,2,3 --num-gpus 4
```

💡 **View results:** Open `comprehensive_analysis_report.html` in any browser

---

### Step 3: Evaluate on New Dataset

```bash
python -m core.evaluate --experiment-id my_test --dataset-csv-path ./data/new_data.csv --sample-size 500 --device-id 0
```

**Required/Important flags:**
- `--experiment-id my_test` - Must match the experiment from Step 2
- `--dataset-csv-path ./data/new_data.csv` - Path to evaluation dataset
- `--device-id 0` - GPU device to use

**Optional flags:**
- `--sample-size 500` - Number of samples to evaluate (default: 5000)
- `--batch-size 64` - Batch size for inference
- `--first-period-truncation true/false` - Truncate at first period

**What happens:**
- Loads the **best classifier** from Step 2 (the optimal layer/hook/token position combination)
- Runs on new dataset (different from training)
- Generates predictions with hallucination probabilities
- Creates evaluation report

**You'll see:**
```
Loading classifier from ./data/classifier/classifier_run_20260216_123456/...
Evaluating: 100%|██████████| 500/500
Accuracy: 0.85, Precision: 0.82, Recall: 0.88
✓ Saved to evaluation_run_20260216_140000/
```

**Outputs:** `./data/classifier/classifier_run_{RUN_ID}/evaluation_run_{TIMESTAMP}/`
- `evaluation_results.csv` - Per-sample predictions with confidence scores
- `evaluation_report.html` - **Open in browser** to view results
- `evaluation_metrics.json` - Accuracy, precision, recall, F1 scores

💡 **Check generalization:** Compare training report (Step 2) vs evaluation report (Step 3)

### Step 4: Interactive Testing (Optional)

```bash
python -m core.real_time_inference --classifier-path ./data/classifier/classifier_run_1705432123/best_classifier_classifier_run_1705432123.pkl --device-id 0
```

**Required flags:**
- `--classifier-path` - Path to saved classifier pkl file from Step 2

**Optional flags:**
- `--device-id 0` - GPU device to use (default: 0)
- `--max-answer-tokens 100` - Max tokens to generate (default: 100)
- `--dataset-csv-path ./data/test.csv` - CSV file for batch mode instead of interactive
- `--debug` - Enable debug logging

**What it does:** Lets you type questions and get real-time hallucination predictions using the best classifier from Step 2.

**Replace:** Both instances of `1705432123` with your actual timestamp from Step 2 (found in the directory name).

## Understanding Outputs

### After Step 2 (Classifier Training)

**Open in browser:** `comprehensive_analysis_report.html`

**What to look for:**
- **Best configuration** - Which layer + hook + token position performed best
- **Confusion matrix** - How many true positives/negatives
- **Performance metrics** - Accuracy, precision, recall, F1 score

**Understanding the results:**
The classifier tested MANY combinations (e.g., 32 layers × 3 hooks × 4 token positions = 384 classifiers). The report shows:
- Which specific **(layer, hook, token position)** combination had the strongest hallucination signal
- Performance of ALL combinations (see `results_all_groups.csv`)
- Only the best-performing classifier is saved as `best_classifier_*.pkl`

**Example:** "Best: Layer 11, hook_z, last_generated (F1: 0.87)" means activations from Layer 11's attention output at the last generated token are most predictive of hallucinations.

**Label meanings:**
- **Label 0** = Correct answer (no hallucination)
- **Label 1** = Incorrect answer (hallucination detected)

### After Step 3 (Evaluation)

**Open in browser:** `evaluation_report.html`

**What to check:**
- **Overall accuracy** - How well it generalizes to new data
- **Precision/Recall** - Trade-off for hallucination detection
- **Sample predictions** - Individual predictions with confidence scores

**Good generalization:** Evaluation metrics close to training metrics (within 5-10%)

## Model Configuration Details

**Which files use the model?**

| File | Model Source | Can Override? |
|------|--------------|---------------|
| `core/generator.py` | `config.py` → `HUGGINGFACE_MODEL_ID` | ✅ Yes, via `--huggingface-model-id` flag |
| `core/classifier.py` | Uses saved activations (model-agnostic) | ❌ No model loaded |
| `core/evaluate.py` | Loads from `config_metadata.json` (auto-detected) | ❌ No override needed |
| `core/real_time_inference.py` | Loads from classifier metadata (auto-detected) | ❌ No override needed |

**Key insight:** Only `generator.py` needs model configuration. Other scripts work with saved data.

**To change model:**
1. Edit `HUGGINGFACE_MODEL_ID` in `config.py`
2. Re-run `python -m core.generator --chunk-id 0 --device-id 0` (Step 1 only)
3. Keep using same classifier/evaluator commands

💡 **Advanced model configuration:** See [docs/DETECTION_ADVANCED.md#model-configuration](docs/DETECTION_ADVANCED.md#model-configuration)

## Common Issues & Quick Fixes

| Problem | Quick Fix | Details |
|---------|-----------|---------|
| **Out of memory** | Reduce `BATCH_SIZE = 4` in config.py | Smaller batches use less VRAM |
| **Import errors** | Run from project root: `cd c:/Users/enqiy/dso-internship-all` | Python needs correct path |
| **"No activations found"** | Check `EXPERIMENT_ID` matches in generator and classifier | Must use same ID |
| **Slow processing** | Reduce layer range: `START_LAYER = 10, END_LAYER = 15` | Fewer layers = faster |
| **Classifier path not found** | Copy exact path from `./data/classifier/` directory | Use full timestamp |

💡 **More issues?** → [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## Complete Minimal Example

**Goal:** Test detection pipeline with 1000 samples

**How it works:** Generator captures activations at all layer/hook/token combinations → Classifier trains and tests all combinations → Best one is saved → Use for evaluation

```bash
# Step 0: Navigate to project root
cd c:/Users/enqiy/dso-internship-all

# Step 1: Edit config.py
# Set: EXPERIMENT_ID = "test"
#      HUGGINGFACE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
#      QA_DATASETS = [{'path': './data/squad_clean.csv', 'samples': 1000}]

# Step 2: Generate activations
python -m core.generator --chunk-id 0 --device-id 0

# Step 3: Train classifier
python -m core.classifier --experiment-id test --gpu-ids 0 --num-gpus 1

# Step 4: View results
# Open: ./data/classifier/classifier_run_*/comprehensive_analysis_report.html

# Step 5: Evaluate on new data
python -m core.evaluate --experiment-id test --dataset-csv-path ./data/new_data.csv --sample-size 500 --device-id 0

# Step 6: View evaluation results
# Open: ./data/classifier/classifier_run_*/evaluation_run_*/evaluation_report.html
```

## Next Steps

**Customize the pipeline:**
- [Change models, hooks, or layers](docs/DETECTION_ADVANCED.md#quick-modification-guide)
- [Understand the hook system](docs/DETECTION_ADVANCED.md#hook-system-explained)
- [Optimize performance](docs/DETECTION_ADVANCED.md#performance-optimization)
- [Extend the pipeline](docs/DETECTION_ADVANCED.md#extending-the-pipeline)

**Get help:**
- [Fix common errors](docs/TROUBLESHOOTING.md)
- [Complete CLI reference](docs/API_REFERENCE.md#detection-pipeline-commands)

