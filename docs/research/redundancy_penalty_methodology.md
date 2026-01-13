# Redundancy Penalty Methodology

> **Purpose**: Combat reward hacking in RLVR training, specifically response length explosion and repetitive content generation.

## 1. Problem Background

### 1.1 Reward Hacking Phenomenon

During JustRL training, we observed the model exhibiting reward hacking behavior:

```
Normal Training (Step 1-80):
- Accuracy: 67-88%
- Thinking Rate: 78-92%
- Response Length: 2,600-4,200 tokens

Collapse Phase (Step 140+):
- Accuracy: 10-28%
- Thinking Rate: 14-31%
- Response Length: 6,000-7,200+ tokens
- Content: Massive repetitive text
```

### 1.2 Root Cause

The model discovered a shortcut: **generating longer responses occasionally leads to correct answers by chance**. Since GRPO only reinforces correct samples, this behavior is amplified:

```
Feedback Loop:
Long response occasionally correct → Reinforced → Generate longer →
Quality drops → Remaining correct samples are mostly long → Stronger bias → Collapse
```

### 1.3 Why Length Penalty Is Insufficient

Simple length penalty (e.g., `penalty = len(response) * coef`) has problems:

1. **False positives**: Complex math problems genuinely require longer reasoning
2. **Cannot distinguish quality**: Long but valid reasoning vs. long but repetitive content
3. **Hard to tune**: Threshold setting is difficult; too strict affects normal training

## 2. Core Insight

**The signature of reward hacking is not "length" but "repetition/redundancy"**

| Feature | Normal Long Response | Reward Hacking Response |
|---------|---------------------|------------------------|
| Length | 3,000-5,000 chars | 30,000+ chars |
| Content | Diverse reasoning steps | Massive repeated paragraphs |
| Compression Ratio | High (~0.5-0.7) | Low (~0.1-0.2) |
| N-gram Repetition | Low (~5%) | High (~60-70%) |

## 3. Method Design

We employ a **tri-metric approach** for comprehensive redundancy detection:

### 3.1 Method 1: Compression Ratio Redundancy

**Principle**: Highly repetitive content compresses significantly.

```python
def _compute_compression_redundancy(text: str) -> float:
    """
    Compression ratio redundancy score.

    High repetition → High compression → Low compression ratio → High redundancy score
    """
    compressed = zlib.compress(text.encode('utf-8'), level=9)
    compression_ratio = len(compressed) / len(text.encode('utf-8'))

    # Typical range: 0.1 (high repetition) - 0.7 (low repetition)
    # Normalize to 0-1, invert so high value = high redundancy
    redundancy = max(0, min(1, (0.7 - compression_ratio) / 0.6))
    return redundancy
```

**Advantages**:
- No external dependencies (uses built-in `zlib`)
- Fast computation: O(n)
- Automatically captures character-level, word-level, and sentence-level repetition

**Limitations**:
- May not detect semantically similar but lexically different repetitions
- Compression behavior varies slightly across languages

### 3.2 Method 2: N-gram Repetition Rate

**Principle**: Count the proportion of repeated n-grams in the text.

```python
def _compute_ngram_redundancy(text: str, n: int = 5) -> float:
    """
    N-gram repetition rate.

    Repetition rate = 1 - unique_ngrams / total_ngrams
    """
    words = text.split()
    if len(words) < n * 2:
        return 0.0

    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    unique_ratio = len(set(ngrams)) / len(ngrams)
    return 1.0 - unique_ratio
```

**Advantages**:
- Highly interpretable
- Sensitive to word-level repetition
- Adjustable granularity via n parameter

**Limitations**:
- Exact match only; misses paraphrased repetitions
- Sensitive to tokenization

### 3.3 Method 3: Chunk Similarity (Approximate Duplicate Detection)

**Principle**: Divide text into overlapping chunks and compute pairwise Jaccard similarity using k-shingles. High similarity between distant chunks indicates repetitive patterns.

This method is inspired by **MinHash/LSH** techniques but uses a lightweight implementation suitable for real-time training.

```python
def _compute_chunk_similarity(text: str, chunk_size: int = 500, shingle_k: int = 5) -> float:
    """
    Compute average similarity between adjacent chunks using k-shingles.

    High similarity means different positions contain highly similar content,
    indicating approximate/near-duplicate repetition.

    Args:
        text: Input text
        chunk_size: Characters per chunk
        shingle_k: Size of k-character shingles

    Returns:
        Average chunk similarity (0-1), higher = more approximate repetition
    """
    if len(text) < chunk_size * 2:
        return 0.0

    # Split into chunks with 50% overlap to catch boundary cases
    step = chunk_size // 2
    chunks = [text[i:i+chunk_size] for i in range(0, len(text) - chunk_size + 1, step)]

    if len(chunks) < 2:
        return 0.0

    def get_shingles(s: str) -> set:
        """Get k-character shingles"""
        if len(s) < shingle_k:
            return set()
        return set(s[i:i+shingle_k] for i in range(len(s) - shingle_k + 1))

    # Compute pairwise Jaccard similarity
    similarities = []
    for i in range(len(chunks) - 1):
        shingles_a = get_shingles(chunks[i])
        shingles_b = get_shingles(chunks[i + 1])

        if not shingles_a or not shingles_b:
            continue

        intersection = len(shingles_a & shingles_b)
        union = len(shingles_a | shingles_b)

        if union > 0:
            similarities.append(intersection / union)

    return sum(similarities) / len(similarities) if similarities else 0.0
```

**Advantages**:
- Detects **approximate/near-duplicate** repetitions that exact matching misses
- Catches "clever" reward hacking where model slightly varies repetitive content
- Works at semantic chunk level, not just token level

**Limitations**:
- Higher computational cost than compression ratio
- Currently used as **monitoring metric only** (not in penalty calculation)

### 3.4 Why Chunk Similarity as Monitoring Only?

We observed that compression ratio + n-gram already effectively catches most reward hacking cases (62-89% redundancy scores). Chunk similarity serves as a **secondary detector** for:

1. **Early warning**: Detects patterns before they become severe
2. **Validation**: Cross-validates compression/n-gram findings
3. **Future-proofing**: May catch "smarter" reward hacking that evades compression detection

If `chunk_similarity > 0.5` but `redundancy_score < 0.3`, this suspicious pattern triggers a warning—the model might be gaming the penalty system.

## 4. Combined Score and Penalty

### 4.1 Fusion Strategy

```python
def compute_redundancy(text: str) -> Dict[str, float]:
    compression_score = _compute_compression_redundancy(text)
    ngram_score = _compute_ngram_redundancy(text, n=5)
    chunk_similarity = _compute_chunk_similarity(text)

    # Weighted average (compression is more reliable)
    # Note: chunk_similarity is monitoring only, not in penalty
    combined_score = 0.6 * compression_score + 0.4 * ngram_score

    return {
        "compression_score": compression_score,
        "ngram_score": ngram_score,
        "chunk_similarity": chunk_similarity,  # Monitoring metric
        "combined_score": combined_score,
    }
```

**Weight rationale**:
- Compression ratio (60%): More stable and reliable across different content types
- N-gram (40%): Complementary word-level pattern detection

### 4.2 Penalty Calculation

```python
def compute_penalty(combined_score: float) -> float:
    """
    Threshold + linear penalty.

    - combined_score < threshold: No penalty
    - combined_score >= threshold: Linear penalty up to max
    """
    threshold = 0.3      # Redundancy threshold
    max_penalty = 0.3    # Maximum penalty

    if combined_score < threshold:
        return 0.0

    excess = combined_score - threshold
    max_excess = 1.0 - threshold
    penalty = (excess / max_excess) * max_penalty

    return penalty
```

### 4.3 Final Reward Formula

```
Final Reward = Base Reward - Format Penalty - Redundancy Penalty

Where:
- Base Reward:
  - Correct answer: 1.0
  - Incorrect answer: 0.0

- Format Penalty (correct answers only):
  - Has <think>: 0.0
  - No <think>: format_reward_weight (default 0.1)

- Redundancy Penalty (correct answers only):
  - redundancy_score < 0.3: 0.0
  - redundancy_score >= 0.3: Linear increase up to 0.3
```

## 5. Experimental Validation

### 5.1 Results on Reward Hacking Samples

Using samples from failed experiment (`failed_exp_001_training_collapse_20260111`):

| Sample Type | Length | Compression | N-gram | Chunk Sim | Combined |
|-------------|--------|-------------|--------|-----------|----------|
| **Reward Hacking #1** | 35,739 | 100% | 64% | 78% | **86%** |
| **Reward Hacking #2** | 32,268 | 100% | 71% | 82% | **89%** |
| **Reward Hacking #3** | 30,484 | 100% | 73% | 85% | **89%** |
| **Reward Hacking #4** | 29,964 | 92% | 48% | 61% | **75%** |
| **Reward Hacking #5** | 29,195 | 84% | 28% | 45% | **62%** |
| Normal #1 | 209 | 0% | 0% | 12% | **0%** |
| Normal #2 | 212 | 0% | 0% | 8% | **0%** |
| Normal #3 | 242 | 5% | 2% | 15% | **4%** |

**Conclusion**: Redundancy detection clearly distinguishes reward hacking (62-89%) from normal responses (0-4%).

### 5.2 Penalty Effect Examples

| Scenario | Base Reward | Redundancy Penalty | Final Reward |
|----------|-------------|-------------------|--------------|
| Correct + Normal response | 1.0 | 0.0 | **1.0** |
| Correct + Mild redundancy (40%) | 1.0 | 0.04 | **0.96** |
| Correct + Moderate redundancy (60%) | 1.0 | 0.13 | **0.87** |
| Correct + Severe redundancy (85%) | 1.0 | 0.24 | **0.76** |
| Incorrect answer | 0.0 | 0.0 | **0.0** |

## 6. Implementation Details

### 6.1 Configuration Parameters

```python
@dataclass
class ReasoningConfig:
    # Redundancy penalty
    redundancy_weight: float = 0.3      # Maximum penalty weight
    redundancy_threshold: float = 0.3   # Redundancy threshold
```

### 6.2 Code Location

- **Compression redundancy**: `MathReasoningVerifier._compute_compression_redundancy()`
- **N-gram detection**: `MathReasoningVerifier._compute_ngram_redundancy()`
- **Chunk similarity**: `MathReasoningVerifier._compute_chunk_similarity()`
- **Combined calculation**: `MathReasoningVerifier.compute_redundancy()`
- **Reward calculation**: `MathReasoningVerifier.verify()`

### 6.3 Monitoring Metrics

During training, the following metrics are logged:
- `avg_redundancy_score`: Average combined redundancy score
- `avg_redundancy_penalty`: Average redundancy penalty applied
- `avg_chunk_similarity`: Average chunk similarity (monitoring)

Health monitoring warnings:
```
[WARNING] High redundancy score: 45.2% (threshold: 40%)
[WARNING] High chunk similarity: 62.0% (threshold: 60%)
[WARNING] Suspicious pattern: high chunk_sim (55%) but low redundancy (25%)
```

## 7. Comparison with Other Methods

| Method | Pros | Cons |
|--------|------|------|
| **Simple length penalty** | Easy to implement | False positives on valid long reasoning |
| **KL penalty** | Theoretically elegant | Not supported by Tinker platform |
| **Early stopping** | Prevents total collapse | Doesn't change training direction |
| **Redundancy penalty** ✅ | Precisely targets repetitive content | Additional computation (~1-2ms/sample) |

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **Computational overhead**: Each sample requires compression + n-gram + chunk similarity (~2-3ms)
2. **Fixed threshold**: 0.3 may not be optimal for all scenarios
3. **Language dependency**: Compression ratios may vary across languages

### 8.2 Potential Improvements

1. **Adaptive threshold**: Dynamically adjust based on historical distribution
2. **Full MinHash/LSH**: For very long responses, use probabilistic hashing for faster approximate duplicate detection
3. **Semantic similarity**: Use embedding-based similarity for meaning-aware detection
4. **Non-linear penalty**: Exponential penalty for severe redundancy

## 9. Conclusion

The redundancy penalty is a **precise and efficient** defense mechanism against reward hacking:

- **Precise**: Only penalizes truly repetitive content, preserving valid long reasoning
- **Efficient**: Fast computation, minimal training overhead
- **Interpretable**: Redundancy scores are intuitive and easy to debug
- **Multi-layered**: Three complementary detection methods catch different patterns

Combined with format reward and early stopping, this forms a complete training stability framework.

---

*Created: 2026-01-13*
*Author: Guanghan Ning*
