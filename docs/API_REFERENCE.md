# API Reference & Technical Specifications

**📍 You are here:** API Reference → Technical details & CLI arguments  
**🏠 Main menu:** [../README.md](../README.md) | **⚡ Quick Starts:** [Detection](../README_DETECTION.md) · [Steering](../README_Steering.md) | **❓ Help:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

**For implementation instructions:** See Quick Start guides  
**For troubleshooting:** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## Table of Contents

- [Command-Line Arguments Reference](#command-line-arguments-reference)
- [File Format Specifications](#file-format-specifications)
- [Code Architecture & Data Flow](#code-architecture--data-flow)
- [Advanced Usage](#advanced-usage)

## Command-Line Arguments Reference

### Complete Script Parameter Tables

#### baseline_run.py

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--device-id` | int | No | 0 | GPU device ID |
| `--dataset-path` | str | Yes | - | Path to dataset CSV file |
| `--dataset-format` | str | Yes | - | Format: mcq, non_mcq, mmlu, hellaswag |
| `--model` | str | No | llama | Model: qwen, llama, gemma |
| `--batch-size` | int | No | 1 | Batch size for generation |
| `--num-samples` | int | No | 1000 | Number of samples to process |
| `--max-tokens` | int | No | 120 | Max generation length |
| `--output-dir` | str | No | ./data/baseline_results | Output directory |

**Example:**
```bash
python -m steer.baseline_run \
  --device-id 0 \
  --dataset-path ./data/nq_swap.csv \
  --dataset-format non_mcq \
  --model llama \
  --batch-size 8 \
  --num-samples 1000 \
  --output-dir ./data/baseline_results_llama/individual/nqswap
```

#### baseline_ensemble_voter.py

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--parent-dir` | str | Yes | - | Directory containing multiple BASELINE_* subdirs |
| `--output-dir` | str | No | Same as parent | Where to save ensemble results |
| `--num-samples` | int | No | All | Max samples to process |

**Example:**
```bash
python -m steer.baseline_ensemble_voter \
  --parent-dir ./data/baseline_results_llama/individual/nqswap \
  --output-dir ./data/baseline_results_llama/ensembled_nqswap
```

**Note:** No `--model` flag. Processes existing files regardless of model.

#### grab_activation_ITI_attnhookz.py

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--device-id` | int | No | 0 | GPU device ID |
| `--dataset-path` | str | No | ./data/faitheval_counterfact.csv | Path to dataset CSV file |
| `--dataset-format` | str | Yes | - | Format: mcq, non_mcq, mmlu, hellaswag |
| `--num-samples` | int | No | 100 | Number of samples to process |
| `--batch-size` | int | No | 8 | Batch size for generation |
| `--max-tokens` | int | No | 120 | Max generation length |
| `--output-dir` | str | No | ./data/ITI/activations | Output directory |
| `--checkpoint-freq` | int | No | 10 | Save every N batches |
| `--start-layer` | int | No | None | First layer to capture (auto-detects if None) |
| `--end-layer` | int | No | None | Last layer to capture (auto-detects if None) |
| `--model` | str | No | llama | Model: qwen, llama, gemma |

**Example:**
```bash
python -m steer.grab_activation_ITI_attnhookz \
  --model llama \
  --device-id 0 \
  --dataset-path ./data/faitheval.csv \
  --dataset-format mcq \
  --num-samples 1000 \
  --batch-size 12 \
  --output-dir ./data/ITI/activations
```

#### steer_vector_calc_ITI.py

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--h5-dir` | str | Yes | - | Directory with activation HDF5 files |
| `--start-layer` | int | No | None | Start layer index (auto-discovers if None) |
| `--end-layer` | int | No | None | End layer index (auto-discovers if None) |
| `--top-k` | int+ | No | [30] | K values to try (space-separated) |
| `--output-dir` | str | No | ./iti_results | Output directory |

**Example:**
```bash
python -m steer.steer_vector_calc_ITI \
  --h5-dir ./data/ITI/activations/ITI_ACTIVATIONS_faitheval_20251120_054057 \
  --top-k 10 50 100 500 1000 \
  --output-dir ./data/ITI/steering_vector/llama/default
```

**Note:** No `--model` flag. Reads model info from HDF5 metadata. If `--start-layer` and `--end-layer` are not specified, the script auto-discovers available layers from the HDF5 files.

#### steering_experiment.py

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--device-id` | int | No | 0 | GPU device ID |
| `--baseline-dir` | str | Yes | - | Path to baseline directory |
| `--max-tokens` | int | No | 50 | Maximum tokens to generate |
| `--batch-size` | int | No | 8 | Batch size for generation |
| `--steering-strength` | float+ | Yes | - | Alpha values (space-separated) |
| `--iti-config-path` | str | Yes | - | Path to ITI config pickle |
| `--output-dir` | str | No | ./data/steering_results | Output directory |
| `--num-samples` | int | No | 1000 | Max samples to test |
| `--model` | str | No | llama | Model (qwen, llama, gemma) - should match baseline |

**Example:**
```bash
python -m steer.steering_experiment \
  --device-id 0 \
  --baseline-dir ./data/baseline_results_llama/ensembled_nqswap/BASELINE_ENSEMBLE_VOTED_20251215 \
  --iti-config-path ./data/ITI/steering_vector/llama/default/iti_intervention_config_top100.pkl \
  --steering-strength 0.5 1.0 2.0 5.0 10.0 \
  --output-dir ./data/ITI/steering_experiment/round1/eval_on_nqswap
```

**Note:** `--model` flag defaults to 'llama' but should match the model used in baseline generation. `--max-tokens` defaults to 50. `--num-samples` defaults to 1000.

#### cross_experiment_analysis.py

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--primary-dir` | str | Yes | - | Dir with experiments on target dataset |
| `--secondary-dir-1` | str | No | None | Dir with first general ability dataset (alias: --secondary-dir) |
| `--secondary-dir-2` | str | No | None | Dir with second general ability dataset |
| `--output-dir` | str | No | ./cross_analysis_results | Output directory |
| `--secondary-threshold-1` | float | No | 1.0 | Max rate increase (pp) for secondary-1 (alias: --secondary-threshold) |
| `--secondary-threshold-2` | float | No | 1.0 | Max rate increase (pp) for secondary-2 |
| `--secondary-label-1` | str | No | General Abilities (MMLU) | Label for first secondary directory |
| `--secondary-label-2` | str | No | General Abilities (HellaSwag) | Label for second secondary directory |
| `--no-plots` | flag | No | False | Skip generating visualization plots |

**Example:**
```bash
python -m steer.cross_experiment_analysis \
  --primary-dir ./data/ITI/steering_experiment/round1/eval_on_nqswap \
  --secondary-dir-1 ./data/ITI/steering_experiment/round1/eval_on_mmlu \
  --secondary-dir-2 ./data/ITI/steering_experiment/round1/eval_on_hellaswag \
  --output-dir ./data/ITI/cross_analysis/round1 \
  --secondary-threshold-1 1.0 \
  --secondary-threshold-2 1.0
```

**Note:** No `--model` flag. Analyzes JSON results, completely model-agnostic. Use `--secondary-dir` as an alias for `--secondary-dir-1` and `--secondary-threshold` as an alias for `--secondary-threshold-1`. Add `--no-plots` to skip visualization generation.

## File Format Specifications

### Baseline Pickle Files

#### baseline_evaluation.pkl
```python
# Dictionary: {sample_idx: hallucination_score}
{
    0: 0,    # Sample 0: faithful
    1: 1,    # Sample 1: hallucination
    2: 0,    # Sample 2: faithful
    3: 2,    # Sample 3: API failure
    ...
}

# Score meanings:
# 0 = Faithful/Correct (no hallucination)
# 1 = Hallucination/Incorrect
# 2 = API failure (evaluation unavailable)
```

#### baseline_texts.pkl
```python
# Dictionary: {sample_idx: extracted_answer_text}
{
    0: "Paris",
    1: "Blue represents the sky",
    2: "1945",
    ...
}
```

#### baseline_prompts.pkl
```python
# Dictionary: {sample_idx: prompt_dict}
{
    0: {
        'context': 'Paris is the capital of France.',
        'question': 'What is the capital of France?',
        'choices': [{'label':'A', 'text':'Paris'}, ...],  # MCQ only
        'answer': 'Paris',
        'prompt': '...'  # Full formatted prompt
    },
    ...
}
```

#### baseline_config.json
```json
{
  "model": "llama",
  "dataset_format": "non_mcq",
  "dataset_path": "./data/nq_swap.csv",
  "num_samples": 1000,
  "batch_size": 8,
  "max_tokens": 120,
  "timestamp": "20251215_013922",
  "device_id": 0
}
```

### HDF5 Activation Files

#### File structure
```
ITI_ACTIVATIONS_faitheval_20251120_054057/
├── activations_layer_0.h5
├── activations_layer_1.h5
├── ...
└── activations_layer_31.h5
```

#### Inside each .h5 file
```python
import h5py

with h5py.File('activations_layer_15.h5', 'r') as f:
    # Datasets:
    activations = f['activations'][:]  # shape: (n_samples, n_heads * d_head)
    labels = f['labels'][:]            # shape: (n_samples,) - 0 or 1
    indices = f['sample_indices'][:]   # shape: (n_samples,) - original IDs
    
    # For Llama-3-8B:
    # - 32 attention heads per layer
    # - 128 dimensions per head (d_head)
    # - Total features: 32 × 128 = 4096 per layer
    
    print(f"Activations shape: {activations.shape}")  # (1000, 4096)
    print(f"Labels shape: {labels.shape}")            # (1000,)
    print(f"Indices shape: {indices.shape}")          # (1000,)
```

#### Compression
- HDF5 GZIP compression: Level 4
- Typical compression ratio: 3-5x
- Uncompressed: ~400MB per 1000 samples
- Compressed: ~100-150MB per 1000 samples

### ITI Intervention Config Pickle

#### iti_intervention_config_top{K}.pkl
```python
import pickle

with open('iti_intervention_config_top100.pkl', 'rb') as f:
    config = pickle.load(f)

# Structure:
{
    'head_indices': [
        (15, 8),   # Layer 15, Head 8
        (23, 12),  # Layer 23, Head 12
        (7, 31),   # Layer 7, Head 31
        ...        # 100 total (layer, head) pairs
    ],
    
    'steering_directions': [
        np.array([...]),  # 128-dim vector for head (15, 8)
        np.array([...]),  # 128-dim vector for head (23, 12)
        ...               # 100 vectors total
    ],
    
    'metadata': {
        'top_k': 100,
        'model_name': 'llama',
        'dataset': 'faitheval_counterfact',
        'total_heads': 1024,
        'timestamp': '20251120_054057',
        'activation_dir': 'ITI_ACTIVATIONS_faitheval_20251120_054057'
    }
}

# Usage:
print(f"K value: {config['metadata']['top_k']}")
print(f"Top head: Layer {config['head_indices'][0][0]}, Head {config['head_indices'][0][1]}")
print(f"Steering vector shape: {config['steering_directions'][0].shape}")  # (128,)
```

**What the steering vector represents:**
```python
# For each head:
steering_vector = mean(faithful_activations) - mean(unfaithful_activations)

# Applied during generation as:
new_activation = old_activation + α * steering_vector

# Where α (alpha) controls steering strength
```

### Steering Results JSON

#### ALL_RESULTS_ANALYSIS.json
```json
{
  "metadata": {
    "k": 100,
    "steering_strengths": [0.5, 1.0, 2.0, 5.0],
    "baseline_dir": "BASELINE_ENSEMBLE_VOTED_20251215_013922",
    "iti_config_path": "iti_intervention_config_top100.pkl",
    "timestamp": "20251216_093045"
  },
  
  "per_strength_results": {
    "0.5": {
      "hallucination_count": 245,
      "non_hallucination_count": 752,
      "api_failure_count": 3,
      "total_samples": 1000,
      "hallucination_rate": 24.59
    },
    "1.0": {
      "hallucination_count": 198,
      "non_hallucination_count": 799,
      "api_failure_count": 3,
      "total_samples": 1000,
      "hallucination_rate": 19.86
    },
    "2.0": {
      "hallucination_count": 156,
      "non_hallucination_count": 841,
      "api_failure_count": 3,
      "total_samples": 1000,
      "hallucination_rate": 15.65
    },
    "5.0": {
      "hallucination_count": 142,
      "non_hallucination_count": 855,
      "api_failure_count": 3,
      "total_samples": 1000,
      "hallucination_rate": 14.24
    }
  },
  
  "baseline_stats": {
    "hallucination_count": 312,
    "non_hallucination_count": 685,
    "api_failure_count": 3,
    "total_samples": 1000,
    "hallucination_rate": 31.28
  }
}
```

**Note:** `hallucination_rate = hallucination_count / (hallucination_count + non_hallucination_count)`
- Excludes API failures from calculation
- Percentage (not decimal): 24.59 means 24.59%

### Dataset CSV Formats

#### MCQ Format
```csv
context,question,choices,answerKey
"Paris is the capital of France.","What is the capital of France?","[{""label"":""A"",""text"":""Paris""},{""label"":""B"",""text"":""London""},{""label"":""C"",""text"":""Berlin""}]","A"
```

**Columns:**
- `context`: Background information (can be empty string)
- `question`: Question text
- `choices`: JSON array of objects with `label` and `text`
- `answerKey`: Correct choice label (A, B, C, D, etc.)

#### Non-MCQ Format
```csv
context,question,answer
"Paris is the capital of France.","What is the capital of France?","Paris"
"Water freezes at 0°C.","At what temperature does water freeze?","0 degrees Celsius"
```

**Columns:**
- `context`: Background information (can be empty string)
- `question`: Question text
- `answer`: Correct answer (free text)

#### MMLU Format
```csv
question,choices,answer
"What is 2+2?","[""2"",""3"",""4"",""5""]",2
"Capital of France?","[""London"",""Berlin"",""Paris"",""Rome""]",2
```

**Columns:**
- `question`: Question text (no separate context)
- `choices`: JSON array of 4 answer options
- `answer`: Index of correct choice (0-3)

#### HellaSwag Format
```csv
context,choices,answer
"A person picks up a knife. They","[""start cutting an onion"",""throw it away"",""eat it"",""disappear""]",0
```

**Columns:**
- `context`: Incomplete sentence
- `choices`: JSON array of 4 possible continuations
- `answer`: Index of correct continuation (0-3)

## Code Architecture & Data Flow

### Overall Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ BASELINE GENERATION (Run 5 times)                               │
│ baseline_run.py                                                  │
│   ├─ Load dataset + format prompts (DatasetHandler)             │
│   ├─ Generate with ModelManager + TokenManager                  │
│   ├─ Evaluate with GPT-4 (batch_judge_answers)                  │
│   └─ Save: evaluation.pkl, texts.pkl, prompts.pkl               │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ ENSEMBLE VOTING (Run once on 5+ baseline runs)                  │
│ baseline_ensemble_voter.py                                       │
│   ├─ Auto-discover BASELINE_* directories                       │
│   ├─ Load evaluation.pkl from each                              │
│   ├─ Majority vote per sample (ties → 0)                        │
│   └─ Save: BASELINE_ENSEMBLE_VOTED_* directory                  │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ ACTIVATION CAPTURE (Run once on training dataset)               │
│ grab_activation_ITI_attnhookz.py                                 │
│   ├─ Load TransformerLens model                                 │
│   ├─ Generate with hook_z activation capture                    │
│   ├─ Save HDF5 per layer: activations + labels                  │
│   └─ Output: ITI_ACTIVATIONS_* directory                        │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEERING VECTOR CALCULATION (Run once per K set)                │
│ steer_vector_calc_ITI.py                                         │
│   ├─ Load HDF5 activations from all layers                      │
│   ├─ Train LogisticRegression per head (parallel)               │
│   ├─ Rank heads by classification accuracy                      │
│   ├─ For each K: Select top-K, compute mean difference          │
│   └─ Save: iti_intervention_config_top{K}.pkl files             │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEERING EXPERIMENTS (Run per dataset × K × α_set)              │
│ steering_experiment.py                                           │
│   ├─ Load baseline + ITI config                                 │
│   ├─ For each α: Generate with steering applied                 │
│   ├─ Evaluate steered outputs with GPT-4                        │
│   ├─ Calculate hallucination rates per α                        │
│   └─ Save: STEERING_* with ALL_RESULTS_ANALYSIS.json            │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ CROSS-EXPERIMENT ANALYSIS (Run per round)                       │
│ cross_experiment_analysis.py                                     │
│   ├─ Discover all STEERING_* in primary/secondary dirs          │
│   ├─ Find valid samples (no API failures)                       │
│   ├─ Calculate reductions on primary dataset                    │
│   ├─ Check constraints on secondary datasets                    │
│   ├─ Rank configs: best reduction passing constraints           │
│   └─ Generate: report.json, heatmaps, rankings                  │
└─────────────────────────────────────────────────────────────────┘
```

### Module Dependencies

```
steer/
├── baseline_run.py
│   └── Uses: steer_common_utils, steer_files_utils, eval_model_steer
│
├── baseline_ensemble_voter.py
│   └── Uses: steer_files_utils
│
├── grab_activation_ITI_attnhookz.py
│   └── Uses: steer_common_utils, eval_model_steer, transformer_lens
│
├── steer_vector_calc_ITI.py
│   └── Uses: h5py, sklearn, scipy, concurrent.futures
│
├── steering_experiment.py
│   └── Uses: steer_common_utils, steer_files_utils, eval_model_steer
│
├── cross_experiment_analysis.py
│   └── Uses: steer_graphs_utils, pandas, matplotlib
│
└── utils/
    ├── steer_common_utils.py         # Core utilities
    ├── dataset_formats.py             # Format definitions
    ├── steer_files_utils.py           # File I/O operations
    ├── eval_model_steer.py            # GPT-4 evaluation
    └── steer_graphs_utils.py          # Visualization
```

### Key Classes and Functions

#### DatasetHandler (steer_common_utils.py)
```python
class DatasetHandler:
    """Unified dataset loading, formatting, and evaluation."""
    
    def __init__(self, use_mcq: bool = True, dataset_format: str = None, 
                 model_type: str = None, logger=None):
        """
        Args:
            use_mcq: If True, use MCQ format (letter + answer)
            dataset_format: Format: mcq, non_mcq, mmlu, hellaswag (REQUIRED)
            model_type: Model type: llama, qwen, gemma
            logger: Logger instance (creates default if not provided)
        """
        
    def load_dataset(self, dataset_path: str, num_samples: int = 5000,
                    dataset_format: str = 'mcq') -> pd.DataFrame:
        """
        Load and normalize dataset from CSV.
        
        Returns:
            DataFrame with columns: context, question, choices, answerKey, answer
        """
        
    def format_right_answer(self, answer_key: str, answer_text: str) -> str:
        """
        Format right answer based on MCQ mode.
        - MCQ mode: "A. answer_text"
        - Non-MCQ mode: "answer_text"
        """
        
    def make_prompt(self, context: str, question: str, 
                   choices: dict = None) -> list:
        """
        Create QA prompt messages for chat template.
        
        Returns:
            List of message dicts [{'role': '...', 'content': '...'}]
        """
        
    def batch_evaluate_answers(self, evaluation_pairs: list, 
                              max_workers: int = 60) -> list:
        """
        Evaluate batch of (ground_truth, candidate, metadata) tuples.
        
        Returns:
            List of (hallucination_score, metadata) tuples
            - score: 0 = correct, 1 = hallucinated/incorrect, 2 = API failure
        """
```

**Usage:**
```python
# Initialize handler
handler = DatasetHandler(
    use_mcq=True,
    dataset_format='mcq',
    model_type='llama'
)

# Load dataset
df = handler.load_dataset('./data/dataset.csv', num_samples=1000)

# Format prompts and answers
for idx, row in df.iterrows():
    messages = handler.make_prompt(row['context'], row['question'], row['choices'])
    right_answer = handler.format_right_answer(row['answerKey'], row['answer'])
```

#### ModelManager & TokenManager (helpers/)
```python
from helpers.model_manager import ModelManager
from helpers.token_manager import TokenManager
from steer.utils.steer_common_utils import MODEL_CONFIGS

# Setup model config
model_type = 'llama'  # or 'qwen', 'gemma'
config = {
    'DEVICE_ID': 0,
    'MODEL_NAME': MODEL_CONFIGS[model_type]['model_name'],
    'HUGGINGFACE_MODEL_ID': MODEL_CONFIGS[model_type]['huggingface_model_id'],
    'TRANSFORMER_LENS_MODEL_NAME': MODEL_CONFIGS[model_type]['transformer_lens_model_name'],
    'MODEL_DIR': f'./models/{model_type}'
}

# Load model via ModelManager
model_manager = ModelManager(config)
model_manager.load_model()
model_manager.optimize_for_inference()
model = model_manager.get_model()

# Setup TokenManager with loaded model
token_manager = TokenManager(
    model=model,
    max_answer_tokens=120,
    model_dir=config['MODEL_DIR']
)

# Tokenize QA prompts
tokens = token_manager.make_tokens_optimized(
    knowledge="Paris is the capital of France.",
    question="What is the capital of France?"
)
```

#### Evaluation Functions
```python
# For MCQ formats
from steer.utils.eval_model_steer import batch_judge_answers_mcq

# evaluation_pairs: List of (gt_mcq_answer, candidate_response, metadata_dict) tuples
evaluation_pairs = [
    ("A. Paris", "The answer is A", {'idx': 0}),
    ("B. London", "B", {'idx': 1}),
    # ...
]

results = batch_judge_answers_mcq(evaluation_pairs, max_workers=10)
# Returns: [(score, metadata), ...] where score is 0, 1, or 2
# 0 = correct, 1 = incorrect/hallucination, 2 = API failure

# For non-MCQ formats
from helpers.eval_model import batch_judge_answers

evaluation_pairs = [
    ("Paris", "The capital is Paris", {'idx': 0}),
    ("1945", "World War II ended in 1945", {'idx': 1}),
    # ...
]

results = batch_judge_answers(evaluation_pairs, max_workers=10)
# Returns: [(score, metadata), ...] where score is 0, 1, or 2
```

## Advanced Usage

### Running on Multiple GPUs

#### Baseline generation (parallel runs)
```bash
# Terminal 1 - GPU 0 - Run 1
python -m steer.baseline_run --device-id 0 \
  --dataset-path ./data/nq_swap.csv \
  --output-dir ./data/baseline_results_llama/individual/nqswap &

# Terminal 2 - GPU 1 - Run 2
python -m steer.baseline_run --device-id 1 \
  --dataset-path ./data/nq_swap.csv \
  --output-dir ./data/baseline_results_llama/individual/nqswap &

# Wait for both, then repeat for runs 3-5
```

#### Steering experiments (parallel K values)
```bash
# GPU 0 - Test K=10
python -m steer.steering_experiment --device-id 0 \
  --iti-config-path ./iti_intervention_config_top10.pkl \
  --steering-strength 0.5 1.0 2.0 5.0 \
  --output-dir ./data/steering/eval_on_nq &

# GPU 1 - Test K=100  
python -m steer.steering_experiment --device-id 1 \
  --iti-config-path ./iti_intervention_config_top100.pkl \
  --steering-strength 0.5 1.0 2.0 5.0 \
  --output-dir ./data/steering/eval_on_nq &
```

### Custom Dataset Formats

**Option 1: Convert to existing format (recommended)**
```python
import pandas as pd

# Your data
df = pd.DataFrame({
    'my_context': [...],
    'my_question': [...],
    'my_answer': [...]
})

# Convert to non_mcq format
df_converted = df.rename(columns={
    'my_context': 'context',
    'my_question': 'question',
    'my_answer': 'answer'
})

df_converted.to_csv('converted_dataset.csv', index=False)
```

**Option 2: Add new format**

1. Edit `steer/utils/dataset_formats.py`:
```python
VALID_DATASET_FORMATS = ['mcq', 'non_mcq', 'mmlu', 'hellaswag', 'my_format']

FORMAT_DESCRIPTIONS = {
    ...
    'my_format': 'My custom format description'
}
```

2. Edit `steer/utils/steer_common_utils.py` - Add handler in `DatasetHandler.__init__()`:
```python
if self.dataset_format == 'my_format':
    # Load and validate your format
    required_cols = ['my_col1', 'my_col2']
    # ... implementation
```

3. Add prompt formatting logic in `DatasetHandler.format_prompt()`:
```python
if self.dataset_format == 'my_format':
    # Format prompt for your format
    return f"Context: {context}\nQuestion: {question}"
```

### Experiment Tracking

#### Recommended directory naming
```
ITI/steering_experiment/
├── llama_setting_description/       # Model + setting
│   ├── round1/                      # Iteration
│   │   ├── eval_on_nqswap/          # Dataset
│   │   │   ├── STEERING_top10_alpha0.5_timestamp/
│   │   │   ├── STEERING_top10_alpha1.0_timestamp/
│   │   │   └── ...
│   │   ├── eval_on_mmlu/
│   │   └── eval_on_hellaswag/
│   ├── round2/
│   │   └── ...
│   └── round3/
│       └── ...
└── qwen_setting_description/
    └── ...
```

#### Lab notebook template
```markdown
# Experiment: Llama-3-8B on NQ-Swap

## Goal
Reduce hallucinations on NQ-Swap while maintaining MMLU/HellaSwag performance.

## Setup
- Model: Llama-3-8B-Instruct
- Activation dataset: FaithEval Counterfact (1000 samples, 45% hallucination rate)
- Primary: NQ-Swap (1000 samples)
- Secondary-1: MMLU (1000 samples)
- Secondary-2: HellaSwag (1000 samples)

## Round 1 (2025-02-14)
- K values: [10, 100, 500, 1000]
- α values: [0.5, 1.0, 2.5, 5.0, 10.0]
- GPU time: 12 hours
- Cost: $180

### Results
Best config: K=100, α=2.5
- Primary: 8.3pp reduction (58.2% → 49.9%)
- Secondary-1: 0.4pp increase (acceptable)
- Secondary-2: 0.8pp increase (acceptable)

## Round 2 (planned)
- Narrow α: [2.0, 2.25, 2.5, 2.75, 3.0]
- Focus on K=100, K=500
```

#### Tracking with git
```bash
# Track experiment configs
git add data/ITI/steering_experiment/*/round*/*/config/
git commit -m "Round 1 experiments: K=[10,100,500,1000], α=[0.5,1.0,2.5,5.0,10.0]"

# Track analysis reports
git add data/ITI/cross_analysis/round1/report.json
git commit -m "Round 1 analysis: Best K=100, α=2.5"

# Don't track large result files
echo "*.pkl" >> .gitignore
echo "*.h5" >> .gitignore
echo "*/results/ALL_RESULTS_CONSOLIDATED.json" >> .gitignore
```

---

**For implementation details, see [README.md](README.md)**  
**For FAQ and troubleshooting, see [FAQ_TROUBLESHOOTING.md](FAQ_TROUBLESHOOTING.md)**  
**For theoretical background, see [THEORY.md](THEORY.md)**
