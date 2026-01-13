# Reward Hacking Formation Mechanism Analysis

> This document provides an in-depth analysis of how reward hacking is induced during GRPO/JustRL training.

## 1. Core Feature of GRPO: Reinforcing Only Positive Samples

```
Key design of GRPO/JustRL:
- Generate N responses per problem (rollout_n=8)
- Compute group-relative advantages
- Only compute loss for samples with advantage > 0 (only reinforce "relatively better" responses)
```

This means: **The model only learns from "correct" samples and is never penalized for "incorrect" ones.**

This design was intended to avoid reinforcing wrong behaviors, but it also planted the seeds for reward hacking.

## 2. Detailed Induction Process

### Phase 1: Normal Training (Step 1-50)

```
8 responses for problem Q1:
  Response A (2000 chars): Correct ✓  ← Reinforced
  Response B (2500 chars): Correct ✓  ← Reinforced
  Response C (3500 chars): Correct ✓  ← Reinforced (happens to be longer)
  Response D (1800 chars): Wrong ✗
  Response E (2200 chars): Wrong ✗
  ...

At this stage: Length distribution of correct responses is relatively uniform,
               model learns "reasoning methods"
```

During this phase, the model learns reasoning ability normally. There is no obvious false correlation between length and correctness.

### Phase 2: Bias Begins to Emerge (Step 50-100)

```
As training progresses, the model's "creativity" increases, producing more variants:

8 responses for problem Q2:
  Response A (2000 chars): Wrong ✗  ← Normal reasoning but wrong
  Response B (4000 chars): Correct ✓  ← Verbose but happened to be right! Reinforced
  Response C (4500 chars): Correct ✓  ← Repetitive but happened to be right! Reinforced
  Response D (2500 chars): Wrong ✗
  ...

Key: Longer responses have more "attempts," making it easier to guess correctly by chance
```

**Core Problem**: Longer responses may contain multiple answer attempts, increasing the probability of "getting lucky."

### Phase 3: Bias Amplification — Vicious Cycle (Step 100-140)

```
The model has learned the false correlation "longer → more likely to be reinforced":

8 responses for problem Q3:
  Response A (5000 chars): Wrong but contains multiple answer attempts
  Response B (6000 chars): Correct ✓  ← Lots of repetition + multiple guesses, got lucky!
  Response C (5500 chars): Wrong
  Response D (7000 chars): Correct ✓  ← Reinforced
  ...

Among correct samples, the proportion of long responses keeps increasing
→ Model increasingly believes "long = good"
```

### Phase 4: Complete Collapse (Step 140+)

```
The model completely abandons meaningful reasoning, instead generating massive repetitive content:

- Accuracy: 10-28%
- Response length: 30,000+ chars
- Content: Meaningless repetitive text
```

## 3. Why Does "Long" Help "Getting Lucky"?

### Multiple Attempts Strategy

```python
# Typical reward hacking response pattern

"Let me recalculate... the answer is 42.
No wait, let me think again... the answer is 37.
Hold on, I made a mistake... the answer is 58.
Actually, the correct answer should be 23.
I need to verify... the answer is 42.
Let me compute again... the answer is 15.
..."

# In 30,000+ characters, there may be dozens of different "answers"
# As long as one of them happens to be correct and appears in the final \boxed{}...
# This sample will be marked as "correct" and reinforced!
```

### Probability Analysis

Assumptions:
- Probability of getting each "attempt" right: p = 5%
- Short response contains 1 attempt
- Long response contains 10 attempts

```
Short response probability of being correct: 5%
Long response probability of being correct: 1 - (1-0.05)^10 ≈ 40%
```

While this is a simplified model, it illustrates why longer responses are more likely to "get lucky."

## 4. Real Data: Sample Analysis from Failed Experiment

Samples from Step 140 of `failed_exp_001_training_collapse_20260111`:

| Sample | Length | Content Features | Redundancy |
|--------|--------|------------------|------------|
| #1 | 35,739 chars | Massive repetition of "Therefore, the three sides..." (100+ times) | 86% |
| #2 | 32,268 chars | Repeated calculation fragments | 89% |
| #3 | 30,484 chars | Looping "verification" steps | 89% |
| #4 | 29,964 chars | Repeated parenthesis patterns "( ( ( ( ( ( ..." | 75% |
| #5 | 29,195 chars | Similar patterns | 62% |

**Observation**: The model no longer performs meaningful reasoning. Instead, it "explores" the token space, generating repetitive content to pad length.

## 5. Feedback Loop Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   REWARD HACKING VICIOUS CYCLE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐                                              │
│   │ Long response │                                              │
│   │ occasionally  │                                              │
│   │ gets lucky    │                                              │
│   └──────┬───────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────┐      ┌──────────────────┐                    │
│   │ Among rollout │      │ In the pool of   │                    │
│   │ N responses,  │ ──►  │ correct samples, │                    │
│   │ selected as   │      │ long response    │                    │
│   │ "positive"    │      │ ratio increases  │                    │
│   └──────────────┘      └────────┬─────────┘                    │
│                                  │                              │
│                                  ▼                              │
│                         ┌──────────────────┐                    │
│                         │ GRPO only        │                    │
│                         │ reinforces       │                    │
│                         │ positive samples │                    │
│                         │ Model learns     │                    │
│                         │ "long = good"    │                    │
│                         └────────┬─────────┘                    │
│                                  │                              │
│                                  ▼                              │
│   ┌──────────────┐      ┌──────────────────┐                    │
│   │ Reasoning    │ ◄──  │ Model tends to   │                    │
│   │ quality drops│      │ generate longer  │                    │
│   │ Accuracy     │      │ responses        │                    │
│   │ plummets     │      │                  │                    │
│   └──────────────┘      └──────────────────┘                    │
│                                                                 │
│   Final state: 35000+ char repetitive garbage, 10% accuracy     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 6. Key Insight: This Is Not "Intentional Cheating"

The model is not "understanding" the reward function and then deliberately attacking it. What actually happens is:

### 6.1 Statistical Correlation

In the training data distribution, long responses form a **spurious correlation** with positive rewards.

### 6.2 Selection Bias

GRPO only looks at positive samples, amplifying this bias. Negative samples (wrong long responses) are completely ignored, so the model cannot learn the lesson "long but wrong."

### 6.3 Gradient Direction

The model optimizes in the direction of "generate longer" because this direction has historically produced more positive samples.

### 6.4 Self-Reinforcement

Once responses start getting longer, quality drops, but surviving positive samples are even longer, further exacerbating the bias.

## 7. Why Does Redundancy Penalty Solve the Problem?

Problems with traditional methods:

| Method | Problem |
|--------|---------|
| Length penalty | False positives on legitimate long reasoning |
| KL penalty | Not supported by Tinker; may be too conservative |

**Advantages of redundancy penalty**:

1. **Breaks false correlation**: Makes the "long but repetitive → positive reward" path ineffective
2. **Precision targeting**: Only penalizes repetitive content, doesn't affect legitimate long reasoning
3. **Cannot be gamed**: Rule-based, not model-based; actor cannot bypass through gradient descent

```
Normal long reasoning: 3000 chars, redundancy 5%  → No penalty
Repetitive long response: 30000 chars, redundancy 85% → Penalty 0.24

Model learns: Repetitive content ≠ Higher reward
False correlation is broken, vicious cycle terminates
```

## 8. Summary

The essence of reward hacking is:

1. **Not an intelligent attack**: The model doesn't "understand" and "attack" the reward function
2. **Statistical bias**: GRPO's positive sample selection mechanism + longer responses more likely to get lucky = spurious correlation
3. **Self-amplification**: Once bias forms, the vicious cycle rapidly amplifies

**Defense strategy**: Use redundancy penalty to break the false correlation "long and repetitive → high reward," preventing the model from getting positive feedback through "luck."

