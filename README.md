# KV Cache Compression System for LLM Inference

Research-quality implementation of novel KV cache eviction policies for long-context LLM inference. This project implements and evaluates attention-score-based eviction, semantic clustering, and learned eviction policies to achieve 50-80% KV cache compression while maintaining <5% perplexity degradation.

## Project Overview

**Goal**: Implement and evaluate novel KV cache eviction policies that reduce memory consumption during long-context LLM inference while maintaining generation quality.

**Key Innovations**:
1. **H2O-style attention-score-based eviction** - Track cumulative attention scores and maintain heavy-hitter pools
2. **Semantic clustering** - Group tokens by key vector similarity for cluster-aware eviction
3. **Learned eviction policies** - Small neural networks predicting token importance

## Environment

- **Platform**: Google Colab (free tier)
- **GPU**: Tesla T4 (16GB VRAM)
- **Model**: Qwen/Qwen2.5-1.5B-Instruct (28 layers, 2 KV heads, 128 head_dim, 128K context)
- **Framework**: PyTorch 2.1+, HuggingFace Transformers 4.40+

## Installation

### For Google Colab

1. Upload `kv_cache_compression.ipynb` to Google Colab
2. Install dependencies by running the first cell (or manually):
   ```bash
   !pip install -r requirements.txt
   ```
3. Run cells sequentially starting from Cell 1

### For Local Development

```bash
pip install -r requirements.txt
jupyter notebook kv_cache_compression.ipynb
```

## Project Structure

The notebook is organized into modular cells:

- **Cell 1**: Environment setup (imports, GPU check, Drive mounting)
- **Cell 2**: Model loading and configuration
- **Cell 3**: Base cache classes (DynamicCache extensions)
- **Cell 4**: Baseline implementations (Sliding Window, LRU)
- **Cell 5-7**: H2O attention-based eviction (MVP)
- **Cell 8-10**: Semantic clustering cache (Intermediate)
- **Cell 11-12**: Learned eviction policy (Stretch goal)
- **Cell 13-15**: Evaluation functions (Perplexity, LongBench, Memory)
- **Cell 16-18**: Benchmarking and visualization
- **Cell 19**: Results saving and checkpointing

## Usage

### Basic Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model (already done in Cell 2)
# model and tokenizer are available

# Use custom cache during generation
from kv_cache import H2OCache

cache = H2OCache(budget=512, heavy_ratio=0.5, recent_ratio=0.5)

outputs = model.generate(
    input_ids,
    past_key_values=cache,
    max_new_tokens=256,
    output_attentions=True
)
```

## Implementation Milestones

### Milestone 1: MVP - H2O Attention-Based Eviction (Weeks 1-5)

- Reference: [H2O (Heavy-Hitter Oracle)](https://arxiv.org/abs/2306.14048)
- Track cumulative attention scores per token
- Maintain heavy-hitter and recent token pools
- Evaluate on WikiText-2 and measure memory compression

### Milestone 2: Intermediate - Semantic Clustering (Weeks 6-8)

- Cluster key vectors using K-means
- Implement cluster-aware eviction strategies
- Compare against H2O at same budget

### Milestone 3: Stretch Goal - Learned Eviction (Weeks 9-10)

- Train lightweight MLP to predict token importance
- Use H2O attention scores as oracle labels
- Compare learned policy against heuristic methods

## Evaluation Metrics

1. **Perplexity** - WikiText-2 sliding window evaluation
2. **Accuracy** - LongBench tasks (NarrativeQA, Qasper, MultiFieldQA)
3. **Memory Usage** - Peak GPU memory and compression ratio
4. **Throughput** - Tokens/second generation speed

## Results

Results will be saved to:
- Google Drive: `/content/drive/MyDrive/kv_cache_compression/results/`
- Local: `./results/` (if not in Colab)

## Key Implementation Details

### Attention Score Extraction

The system uses `output_attentions=True` during generation to track attention weights. For T4 GPUs, we use SDPA (scaled dot product attention) which supports attention output, avoiding Flash Attention 2 incompatibility.

### Memory-Efficient Eviction

Uses `torch.index_select` with sorted indices to avoid creating unnecessary tensor copies during eviction.

### Position Embedding Handling

After eviction, position IDs are handled correctly using the `cache_position` parameter in Transformers 4.36+.

## References

- [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)
- [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://github.com/THUDM/LongBench)

## License

This project is for research purposes as part of a Master's thesis.

## Author

Research implementation for Master's thesis project on KV cache compression for LLM inference.





