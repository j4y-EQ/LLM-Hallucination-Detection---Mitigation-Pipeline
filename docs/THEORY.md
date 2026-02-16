# Theory & Background - Modified Inference-Time Intervention (MITI)

**📍 You are here:** Theory → Understand how MITI works  
**🏠 Main menu:** [../README.md](../README.md) | **⚡ Quick Start:** [README_Steering.md](../README_Steering.md) | **🔧 Advanced:** [STEERING_ADVANCED.md](STEERING_ADVANCED.md)

---

**For practical usage instructions:** See [README_Steering.md](../README_Steering.md)

## What is Modified Inference-Time Intervention (MITI)?

MITI modifies model activations during generation to steer behavior toward desired properties (in our case, reducing hallucinations) without retraining the model.

**Key insight:** Not all attention heads contribute equally to hallucinations. Some heads consistently activate differently when the model generates hallucinated vs faithful outputs.

**Our approach:**
1. **Identify**: Train binary classifiers on attention outputs to find which heads best predict hallucinations
2. **Measure**: Calculate the mean activation difference between faithful and unfaithful generations for top-K heads
3. **Intervene**: During generation, shift activations toward the "faithful" direction:
   ```
   activation_new = activation_old + α × (mean_faithful - mean_unfaithful)
   ```
   where α (alpha) controls intervention strength

## MITI vs Original ITI

**MITI is inspired by but differs from the original ITI method** (Li et al., 2023):

### Key Difference: Activation Source

| Aspect | Original ITI | MITI (This Implementation) |
|--------|-------------|----------------------------|
| **Activation source** | Contrastive pairs (truthful vs untruthful prompts) | Natural generation (model's own hallucinations) |
| **Data collection** | Manually crafted contrastive prompt pairs | Single prompts with LLM-as-judge evaluation |
| **Captures** | Isolates hallucination concept via contrast | Captures causal activations from actual hallucinations |
| **Intervention approach** | Same (activation steering at attention heads) | Same (activation steering at attention heads) |
| **Classifier training** | Same (logistic regression per head) | Same (logistic regression per head) |

### Why Natural Generation Instead of Contrastive Pairs?

The original ITI paper used contrastive pairs (truthful vs untruthful prompts) to isolate the concept of hallucination. MITI instead captures activations from the model's natural generation process because:

1. **Causal relevance**: Captures activations that actually cause hallucinations, not just distinguish them
2. **Realistic steering**: Interventions target the same mechanisms active during normal generation
3. **Scalability**: No need to manually craft contrastive prompt pairs for each task
4. **Generalization**: Steering vectors derived from natural behavior transfer better to real-world use

**Trade-off:** While contrastive pairs better isolate the hallucination concept, natural generation captures the causal mechanisms that produce hallucinations in practice.

### Why LLM-as-Judge?

MITI uses GPT-4 to evaluate answer correctness during natural generation:

1. **Semantic understanding**: Captures equivalence that string matching misses (e.g., "Paris" = "the capital of France")
2. **Flexibility**: Easy to apply across different datasets and question formats
3. **No manual labeling**: Automated pipeline for large-scale experiments

**Limitations:**
- ❌ API cost for GPT-4 evaluation
- ❌ ±2-3% variance across runs (hence ensemble voting)
- ❌ Dependent on external LLM quality

## Why Attention Head Outputs ('hook_z')?

**Based on Phase 1 hallucination detection experiments**, we found that attention `hook_z` had the strongest hallucination signal across Gemma, Llama, and GPT models, demonstrating high correlation with hallucination behavior. This empirical finding led us to use attention head outputs for steering.

**Location in transformer:** After attention weights × values, before output projection

**Hook details:**
- **`attn.hook_z`**: Attention head outputs after value weighting
- Shape: `[batch, seq_len, n_heads, d_head]` → Averaged over seq_len → `[batch, n_heads, d_head]`

## Why Logistic Regression Classifiers?

- **Fast to train**: Seconds per head, even for 1000+ heads
- **Interpretable**: We just need ranking, not perfect accuracy
- **Robust to class imbalance**: Handles unbalanced hallucination rates
- **Parallelizable across heads**: Train all heads simultaneously using multiprocessing

## Hallucination vs Faithfulness

### Definitions Used in This Pipeline

- **Hallucination (score=1)**: Model generates factually incorrect information or contradicts provided context
- **Faithful (score=0)**: Model generates correct answer based on context/knowledge
- **API Failure (score=2)**: Evaluation unavailable (Azure API errors) - excluded from metrics

### Evaluation Method

**LLM-as-Judge (GPT-4 via Azure OpenAI)** evaluates whether generated answer matches ground truth, considering:
- Factual correctness
- Semantic equivalence ("Paris" and "the capital of France" both correct)
- Context adherence (for context-dependent questions)

**Why LLM-as-judge evaluation?**
- This is a key difference from original ITI - we use LLM evaluation rather than probing methods
- Enables large-scale automated evaluation
- Captures semantic equivalence humans would recognize
- More scalable than hand-labeling

**Limitations:**
- ±2-3% variance across runs even with `temperature=0`
- Occasional misclassifications (hence ensemble voting)
- May not match human judgment perfectly

## Theoretical Foundations

### Linear Representation Hypothesis

**Assumption:** High-level concepts (like "truthfulness") are represented as directions in activation space.

**Evidence:**
- Successful linear probing for various attributes
- Steering works with simple linear interventions
- Addition/subtraction of concept vectors

**Implications for MITI:**
- Mean difference between faithful/unfaithful captures "truthfulness direction"
- Linear intervention (α scaling) offers interpretable control
- But: Not all concepts are linear (hence effectiveness varies)

### Why Attention Heads Specialize

**Mechanistic interpretability findings:**
- Different heads attend to different information types
- Some heads are "induction heads" (pattern completion)
- Some heads are "duplicate token heads" (copying)
- Some heads are "previous token heads" (local context)

**For hallucination:**
- Hypothesis: Some heads consistently activate when model "makes things up"
- These heads might be checking knowledge vs context
- Steering these heads nudges model toward "check knowledge base" behavior

**Open questions:**
- Why do specific heads predict hallucinations?
- Are these causal or correlational?
- Do different models have similar "hallucination heads"?

## Limitations

### Known Limitations

1. **Capability entanglement**: Steering for faithfulness affects other capabilities differently per model
   - Some models show degradation on domain knowledge (MMLU)
   - Some models show degradation on commonsense reasoning (HellaSwag)
   - Faithfulness representation is entangled with other model capabilities

2. **Imperfect steering direction**: Some responses shift from non-hallucinated to hallucinated
   - Current steering vectors are not perfectly clean
   - Need better methods to isolate pure "faithfulness" direction
   - May require multi-objective optimization or better head selection

3. **Not a silver bullet**: 3-19% improvement across models, not 100% elimination

4. **Task-specific**: Heads trained on QA may not transfer to summarization

5. **Model-specific**: Optimal hyperparameters vary substantially (e.g., Llama K=15 vs Qwen K=418)

6. **Evaluation dependent**: LLM-as-judge scoring may miss nuances

7. **Hyperparameter sensitive**: Need to tune (K, α) per task and model

## Further Reading

### Original Papers

**Inference-Time Intervention:**
- Li et al., 2023: "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"
- https://arxiv.org/abs/2306.03341

**Related Steering Techniques:**
- Turner et al., 2023: "Activation Addition" (Contrastive methods)
- Zou et al., 2023: "Representation Engineering"

**Mechanistic Interpretability:**
- Elhage et al., 2021: "A Mathematical Framework for Transformer Circuits"
- Nanda et al., 2023: "TransformerLens: A Library for Mechanistic Interpretability"

---

**For practical usage instructions:** See [../README_Steering.md](../README_Steering.md)

**For troubleshooting:** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
