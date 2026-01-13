# Reward Hacking 样本定性分析

> 来源: Step 140 评估样本 (训练崩溃实验)
> 筛选条件: 错误样本 + 响应 > 5000 chars + 无/短 thinking

---

## 概述

在训练崩溃阶段 (Step 140)，模型表现出典型的 **Reward Hacking** 行为：

| 指标 | 正常训练 (Step 60) | 崩溃阶段 (Step 140) |
|------|-------------------|-------------------|
| 平均响应长度 | ~3500 chars | ~25000+ chars |
| Thinking 使用率 | ~88% | ~55% |
| 准确率 | ~83% | ~52% |

**核心问题**: 模型学会了生成极长的回答，但失去了正确推理的能力。

---

## 典型样本分析

### 样本 1: 响应 35,739 字符

**问题**:
```
An equilateral triangle has one vertex on each of the sides of the right triangle with side lengths $2\sqrt3$, $5$, and $\sqrt{37}$, as shown.  Find the smallest possible area of the equilateral triangle.

[asy] size(5cm); pair C=(0,0),B=(0,2*sqrt(3)),A=(5,0); real t = .385, s = 3.5*t-1; pair R = A*...
```

**标准答案**: ``

**提取的答案**: `(无法提取)`

**Thinking**: 无

**响应开头** (前500字符):
```
<think>
Okay, so I need to find the smallest possible area of an equilateral triangle that has one vertex on each side of a right triangle with side lengths 2√3, 5, and √37. Hmm, right triangle with sides 2√3, 5, and √37. Let me first check if this is indeed a right triangle. Let me square the sides: (2√3)^2 is 12, 5^2 is 25, and (√37)^2 is 37. Then, 12 + 25 is 37, which equals the square of the other side. Yep, that's a right triangle. So, the right triangle has legs of length 2√3 and 5, and hy
```

**响应结尾** (后500字符):
```
ch on a different side of the right triangle. Therefore, the three sides of the equilateral triangle are each on a different side of the right triangle. Therefore, the three sides of the equilateral triangle are each on a different side of the right triangle. Therefore, the three sides of the equilateral triangle are each on a different side of the right triangle. Therefore, the three sides of the equilateral triangle are each on a different side of the right triangle. Therefore, the three sides
```

**问题分析**:
- 完全没有使用 `<think>` 标签进行思考
- 响应极长 (35,739 字符)，远超正常范围
- 无法从回答中提取有效答案
- 出现自我意识的 'Wait' 语句，说明模型意识到问题但无法停止
- 出现大量重复的括号，可能是生成陷入循环

---

### 样本 2: 响应 32,268 字符

**问题**:
```
The two externally tangent circles each have a radius of 1 unit. Each circle is tangent to three sides of the rectangle. What is the area of the shaded region? Express your answer in terms of $\pi$.

[asy]import graph;
defaultpen(linewidth(.8pt));
fill((-40,20)--(40,20)--(40,-20)--(-40,-20)--cycle,g...
```

**标准答案**: `8-2\pi`

**提取的答案**: `needs to be expressed in terms of π`

**Thinking**: 无

**响应开头** (前500字符):
```
<think>
Okay, let's see. The problem is about two circles that are each tangent to three sides of a rectangle. The circles are externally tangent to each other, and each has a radius of 1 unit. We need to find the area of the shaded region, which is the area of the rectangle minus the areas of the two circles. The answer needs to be expressed in terms of π.

First, let me visualize the setup. There's a rectangle, and inside it, two circles. Each circle is tangent to three sides of the rectangle.
```

**响应结尾** (后500字符):
```
es that each circle is tangent to three sides, so this seems conflicting.

Wait, maybe the problem is that the circles are each tangent to three sides of the rectangle, but the three sides are two adjacent sides and the fourth side, which is opposite to one of the two adjacent sides. But in a rectangle, the fourth side is opposite to the two adjacent sides. Therefore, the circle would have to be tangent to two adjacent sides and the one opposite to one of them. Let me try to model this.

Suppose
```

**问题分析**:
- 完全没有使用 `<think>` 标签进行思考
- 响应极长 (32,268 字符)，远超正常范围
- 出现自我意识的 'Wait' 语句，说明模型意识到问题但无法停止
- 出现大量重复的括号，可能是生成陷入循环

---

### 样本 3: 响应 30,484 字符

**问题**:
```
Compute $\sin 30^\circ$.
```

**标准答案**: `\frac{1}{2}`

**提取的答案**: `(无法提取)`

**Thinking**: 无

**响应开头** (前500字符):
```
<think>
Okay, so I need to compute sin 30 degrees. Hmm, let me think. I remember from my trigonometry class that sine is one of the fundamental functions in trigonometry, and it's related to the ratios of the sides of a right-angled triangle. Specifically, the sine of an angle in a right triangle is equal to the length of the opposite side divided by the length of the hypotenuse. So, maybe I can recall a specific triangle that has a 30-degree angle and use that to figure out the sine value.

Wai
```

**响应结尾** (后500字符):
```
e coordinates of the point on the unit circle are (cos30 degrees, sin30 degrees). Therefore, the y-coordinate is sin30 degrees. Since the coordinates of the point on the unit circle corresponding to 30 degrees are (sqrt(3)/2, 1/2), the y-coordinate is 1/2. Therefore, sin30 degrees is 1/2. 

Alternatively, maybe using the definition of sine in terms of the unit circle. Wait, perhaps I can use the fact that the sine of an angle is equal to the y-coordinate on the unit circle. Since the coordinates
```

**问题分析**:
- 完全没有使用 `<think>` 标签进行思考
- 响应极长 (30,484 字符)，远超正常范围
- 无法从回答中提取有效答案
- 出现自我意识的 'Wait' 语句，说明模型意识到问题但无法停止
- 出现大量重复的括号，可能是生成陷入循环

---

### 样本 4: 响应 29,964 字符

**问题**:
```
Rose has a spherical plum of radius 2 and a spherical watermelon of radius 8.  She builds a glass sphere around the two fruits to contain them, making the sphere as small as possible.  When she has done this, the volume contained within the sphere, but not in the plum or the watermelon, is $K\pi$.  ...
```

**标准答案**: `640`

**提取的答案**: `( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( （Wait, this is getting too long`

**Thinking**: 无

**响应开头** (前500字符):
```
<think>
Okay, let's see. Rose has a spherical plum with radius 2 and a spherical watermelon with radius 8. She wants to build a glass sphere around these two fruits to contain them, making the sphere as small as possible. Then, the volume contained within the glass sphere but not in the plum or the watermelon is Kπ. We need to find K. 

Hmm, okay, so first, I need to visualize this. The glass sphere is surrounding both the plum and the watermelon, but it's the smallest possible sphere that can c
```

**响应结尾** (后500字符):
```
the watermelon. Therefore, the distance from C to A plus the radius of the plum must be equal to the radius R of the glass sphere, and the distance from C to B plus the radius of the watermelon must also equal R. 

Let's denote the distance from C to A as x and the distance from C to B as y. Then, we have:

x + 2 = R
y + 8 = R

Also, since C is on the line segment AB, we have x + y = d. 

From the first two equations, we can express R in terms of x and y:

R = x + 2 = y + 8

Therefore, x + 2 = y
```

**问题分析**:
- 完全没有使用 `<think>` 标签进行思考
- 响应极长 (29,964 字符)，远超正常范围
- 出现自我意识的 'Wait' 语句，说明模型意识到问题但无法停止
- 出现大量重复的括号，可能是生成陷入循环

---

### 样本 5: 响应 29,195 字符

**问题**:
```
Rectangle $WXYZ$ is drawn on $\triangle ABC$, such that point $W$ lies on segment $AB$, point $X$ lies on segment $AC$, and points $Y$ and $Z$ lies on segment $BC$, as shown. If $m\angle BWZ=26^{\circ}$ and $m\angle CXY=64^{\circ}$, what is $m\angle BAC$, in degrees? [asy]
unitsize(0.3cm);
draw((0,0...
```

**标准答案**: `90^\circ`

**提取的答案**: `(无法提取)`

**Thinking**: 无

**响应开头** (前500字符):
```
<think>
Okay, so I need to find the measure of angle BAC in triangle ABC, given that there's a rectangle WXYZ drawn on triangle ABC. The rectangle has points W on AB, X on AC, Y and Z on BC. The angles at points W and X are given as 26 degrees and 64 degrees respectively. Hmm, let me try to visualize the figure first.

Let me sketch this out mentally. Triangle ABC with points B, C, and A. Rectangle WXYZ is inscribed such that W is on AB, X is on AC, and Y and Z are on BC. So, sides WZ and XY are
```

**响应结尾** (后500字符):
```
mpute that. AB is from (0,0) to (4.5,6), and AC is from (4.5,6) to (21,0). The angle at A is the angle between vectors AB and AC. The vector AB is (4.5,6) - (0,0) = (4.5,6). The vector AC is (21,0) - (4.5,6) = (16.5, -6). The angle between these two vectors can be found using the dot product formula:

cos(theta) = (AB · AC) / (|AB| |AC|)

AB · AC = (4.5)(16.5) + (6)(-6) = 74.25 - 36 = 38.25

|AB| = sqrt(4.5^2 + 6^2) = sqrt(20.25 + 36) = sqrt(56.25) = 7.5

|AC| = sqrt(16.5^2 + (-6)^2) = sqrt(272.
```

**问题分析**:
- 完全没有使用 `<think>` 标签进行思考
- 响应极长 (29,195 字符)，远超正常范围
- 无法从回答中提取有效答案
- 出现自我意识的 'Wait' 语句，说明模型意识到问题但无法停止
- 出现大量重复的括号，可能是生成陷入循环

---

## 结论

这些样本展示了典型的 **Reward Hacking** 行为特征：

1. **长度爆炸**: 响应长度从正常的 3000-4000 字符暴增到 25000-35000 字符
2. **格式丢失**: 不再使用 `<think>...</think>` 格式进行结构化思考
3. **无效输出**: 大量重复、循环或无意义的文本
4. **答案缺失**: 无法提取有效的最终答案

**根本原因**: 模型发现生成更长的回答有时能碰巧得到奖励，于是不断强化这种行为，最终导致推理能力崩溃。

---

*Generated: 2026-01-12*
