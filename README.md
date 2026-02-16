# LLM Hallucination Detection & Steering

Two pipelines for analysing and reducing hallucinations in Large Language Models (LLMs).

---

### **Documentation Structure (Quick Reference)**

```
📁 Root Directory
│
├─ 📄 README.md ...................... ← YOU ARE HERE (start point)
├─ 📄 README_DETECTION.md ............  Run detection (3 steps)
└─ 📄 README_Steering.md .............  Run steering (6 steps)

📁 docs/ (Advanced Topics)
│
├─ 📄 DETECTION_ADVANCED.md ..........  Customize detection
├─ 📄 STEERING_ADVANCED.md ...........  Customize steering
├─ 📄 TROUBLESHOOTING.md .............  Fix problems
├─ 📄 THEORY.md ......................  Understand MITI
└─ 📄 API_REFERENCE.md ...............  Technical specs
```
---

## 🚀 START HERE: Installation

### Prerequisites

**Required for both pipelines:**
- **Python 3.10**
- **Azure OpenAI API access** (for GPT-4 hallucination evaluation - REQUIRED)
- QA dataset CSV files

---

## Installation Steps

### 1. Install Dependencies

```bash
cd c:/Users/enqiy/dso-internship-all
pip install -r requirements.txt
```

### 2. Configure API Keys (Required for Both Pipelines)

**Create `.env` file in the project root directory** (`c:/Users/enqiy/dso-internship-all/.env`):

```bash
# Azure OpenAI Configuration (REQUIRED for hallucination evaluation)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
EVAL_MODEL=gpt-4o

# HuggingFace Token (optional, for gated models like Llama)
HF_TOKEN=your_huggingface_token
```


**Important:** Both detection and steering pipelines use GPT-4 for hallucination evaluation.

---

## ✅ Installation Complete! Now Choose Your Pipeline

### Detection Pipeline

**What it does:** Captures internal model activations during text generation, trains classifiers to detect hallucination patterns, and evaluates performance on new datasets.

**Ready to run:** **[README_DETECTION.md](README_DETECTION.md)** - 3 steps to get started

---

### Steering Pipeline

**What it does:** Identifies attention heads that contribute to hallucinations, applies steering vectors during generation to reduce hallucination rates in real-time.

**Ready to run:** **[README_Steering.md](README_Steering.md)** - 6 steps to get started

**Understand the theory:** **[docs/THEORY.md](docs/THEORY.md)** - How MITI works

---

## 📖 How to Navigate This Documentation

### **I'm a First-Time User - Where Do I Start?**

```
Step 1: You're already here! (README.md) ✅
        ↓
Step 2: Complete Installation above ⬆️ (install dependencies + create .env file)
        ↓
Step 3: Choose Detection or Steering pipeline above ⬆️
        ↓
Step 4: Click the README link for your chosen pipeline
        ↓
Step 5: Follow the Quick Start guide (3-6 steps)
        ↓
Step 6: Run the commands → Get results!
```

**Example path for Detection:**
```
README.md → Install + .env (done) ✅ → README_DETECTION.md → Run 3 commands → View HTML report → Done!
```

**Example path for Steering:**
```
README.md → Install + .env (done) ✅ → README_Steering.md → Run 6 steps → Check results → Done!
```

---

## Project Structure

```
dso-internship-all/
│
├── README.md                    # ← Start here
├── README_DETECTION.md          # Detection pipeline (3 steps)
├── README_Steering.md           # Steering pipeline (6 steps)
│
├── docs/                        # Advanced guides
│   ├── DETECTION_ADVANCED.md
│   ├── STEERING_ADVANCED.md
│   ├── TROUBLESHOOTING.md
│   ├── THEORY.md
│   └── API_REFERENCE.md
│
├── core/                        # Detection code
│   ├── generator.py
│   ├── classifier.py
│   └── evaluate.py
│
├── steer/                       # Steering code
│   ├── baseline_run.py
│   ├── grab_activation_ITI_attnhookz.py
│   └── steering_experiment.py
│
├── helpers/                     # Shared utilities
├── config.py                    # Configuration
└── data/                        # Generated outputs (created on first run)
```

---

### **I Want to Customize - Where Do I Go?**

**After completing the Quick Start, go here:**

| What You Want to Change | File to Read | Section |
|-------------------------|--------------|---------|
| **Detection:** Change model | [docs/DETECTION_ADVANCED.md](docs/DETECTION_ADVANCED.md) | Model Configuration |
| **Detection:** Change layers/hooks | [docs/DETECTION_ADVANCED.md](docs/DETECTION_ADVANCED.md) | Hook System |
| **Steering:** Add new model | [docs/STEERING_ADVANCED.md](docs/STEERING_ADVANCED.md) | Adding New Models |
| **Steering:** Custom dataset format | [docs/STEERING_ADVANCED.md](docs/STEERING_ADVANCED.md) | Custom Dataset Formats |
| **Steering:** Understand K and α | [docs/STEERING_ADVANCED.md](docs/STEERING_ADVANCED.md) | Parameter Selection |
| Fix errors | [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common Issues |
| Understand MITI theory | [docs/THEORY.md](docs/THEORY.md) | Full document |
| Complete CLI reference | [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | Command arguments |

---

## Next Steps

✅ **Installation complete** (dependencies + .env file set up above)

**Now choose your pipeline:**
- 🔍 **[Detection Quick Start](README_DETECTION.md)** - 3 commands to detect hallucinations
- 🎯 **[Steering Quick Start](README_Steering.md)** - 6 steps to reduce hallucinations


## Support & Help

**Having issues?**
1. Check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common problems
2. Verify you're running from project root: `cd c:/Users/enqiy/dso-internship-all`
3. Ensure correct Python module syntax: `python -m core.generator` (not `python core/generator.py`)

**Need clarification?**
- Quick Start guides include "Common Issues" sections
- Advanced guides have detailed examples and explanations
