# Propensity Score Trading - Explained Simply

## What is This About?

Imagine you're trying to figure out if eating breakfast makes you do better on tests. You notice that students who eat breakfast often score higher. But wait - maybe those students also sleep more, have more supportive parents, or study harder. How do you know if breakfast *actually* helps, or if it's just these other things?

**Propensity Score Methods** help us answer this exact question in trading: "Does my trading signal *actually* cause better returns, or is something else going on?"

---

## The Real-World Problem

### Think of it Like This:

Let's say you have a trading strategy: "Buy stocks when their price crosses above the 50-day average."

You test it and see great returns! But here's the catch:
- This signal tends to trigger more often when the overall market is calm (low volatility)
- Calm markets generally have better returns anyway
- So... did your signal work, or was it just lucky timing?

It's like claiming "umbrellas cause rain" because you see umbrellas whenever it rains. The truth is: weather conditions (a *confounding factor*) affect both!

```
Weather Conditions → People carry umbrellas
Weather Conditions → Rain happens

NOT: Umbrellas → Rain!
```

In trading:
```
Market Conditions → Your signal fires
Market Conditions → Returns are good

Your signal alone? Maybe not the cause!
```

---

## How Propensity Scores Work

### The Simple Idea

**Propensity Score** = The probability that your trading signal would fire, given all the market conditions at that moment.

Think of it as asking: "How likely was this signal to appear in this specific situation?"

### The Matching Game

Here's the clever trick:

1. **Calculate Propensity Scores**: For every moment in time, calculate how likely your signal was to fire based on market conditions (volatility, trend, volume, etc.)

2. **Find Twins**: Match moments when your signal DID fire with similar moments when it DIDN'T fire, based on similar propensity scores

3. **Compare Fairly**: Now compare returns between these matched pairs

It's like finding two almost identical days - same volatility, same trend, same volume - but one day your signal said "buy" and the other day it didn't. Now you can fairly compare!

---

## A School Example

Imagine you're the school principal trying to figure out if a tutoring program helps students.

**The Problem**: Students who choose tutoring might already be more motivated, have supportive parents, or start with better grades. Just comparing tutored vs non-tutored students isn't fair!

**The Solution**:
1. Calculate each student's "propensity" to join tutoring based on: motivation score, parent involvement, starting grades, etc.
2. Find pairs of students with similar propensity scores - one who joined tutoring, one who didn't
3. Compare their final grades

Now you're comparing apples to apples!

---

## Trading Example: Simple Walkthrough

### Scenario: Testing a Momentum Signal

You have a signal: "Buy when 5-day return > 2%"

**Step 1: Gather Data**
For each trading day, record:
- Did the signal fire? (Yes/No)
- Market volatility that day
- Trading volume
- Overall market trend
- The next day's return

**Step 2: Calculate Propensity Scores**

Use machine learning to predict: "Given these market conditions, how likely is the signal to fire?"

Example results:
| Day | Signal Fired | Volatility | Volume | Propensity Score |
|-----|--------------|------------|--------|------------------|
| 1   | Yes          | Low        | High   | 0.75             |
| 2   | No           | Low        | High   | 0.73             |
| 3   | Yes          | High       | Low    | 0.25             |
| 4   | No           | High       | Low    | 0.28             |
| 5   | Yes          | Medium     | Medium | 0.50             |

**Step 3: Match Similar Days**

- Day 1 (signal: Yes, PS: 0.75) matches with Day 2 (signal: No, PS: 0.73)
- Day 3 (signal: Yes, PS: 0.25) matches with Day 4 (signal: No, PS: 0.28)

**Step 4: Compare Returns**

| Match | Signal Day Return | No-Signal Day Return | Difference |
|-------|-------------------|----------------------|------------|
| 1-2   | +1.5%             | +0.8%                | +0.7%      |
| 3-4   | -2.0%             | -2.5%                | +0.5%      |

**Average Causal Effect**: +0.6% (your signal adds 0.6% on average!)

---

## Why This Matters for Trading

### Without Propensity Scores:
"My signal made 20% this year!"
(But maybe the market also went up 18%... so your signal only really added 2%)

### With Propensity Scores:
"After controlling for market conditions, my signal adds 0.5% per trade on average."
(This is the TRUE causal effect!)

---

## Three Common Techniques

### 1. Matching (Finding Twins)
- Find similar days, compare outcomes
- Like comparing identical twins who made different choices

### 2. Weighting (Adjusting Importance)
- Give more weight to rare events
- If your signal rarely fires in high-volatility markets, those rare moments get more weight

### 3. Stratification (Group Comparison)
- Divide data into groups (buckets) by propensity score
- Compare within each group
- Like comparing students within the same grade level

---

## Real Code Example (Simplified Python)

```python
# Super simplified example

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# Your trading data
data = pd.DataFrame({
    'signal_fired': [1, 0, 1, 0, 1, 0, 1, 0],  # 1 = signal fired
    'volatility': [0.1, 0.12, 0.3, 0.28, 0.15, 0.14, 0.2, 0.22],
    'volume': [100, 95, 50, 55, 80, 85, 70, 68],
    'next_day_return': [0.02, 0.01, -0.03, -0.04, 0.015, 0.008, 0.01, 0.005]
})

# Step 1: Calculate propensity scores
model = LogisticRegression()
features = data[['volatility', 'volume']]
model.fit(features, data['signal_fired'])
data['propensity_score'] = model.predict_proba(features)[:, 1]

# Step 2: Match treated with untreated
treated = data[data['signal_fired'] == 1]
untreated = data[data['signal_fired'] == 0]

# Find nearest matches based on propensity score
nn = NearestNeighbors(n_neighbors=1)
nn.fit(untreated[['propensity_score']])
distances, indices = nn.kneighbors(treated[['propensity_score']])

# Step 3: Calculate average treatment effect
matched_untreated = untreated.iloc[indices.flatten()]
ate = treated['next_day_return'].mean() - matched_untreated['next_day_return'].mean()

print(f"Average Treatment Effect (ATE): {ate:.4f}")
print(f"This means: Your signal adds approximately {ate*100:.2f}% per trade")
```

---

## Common Mistakes to Avoid

### 1. Using Future Information
Bad: Including tomorrow's price to calculate today's propensity score
Good: Only use information available at decision time

### 2. Ignoring Important Confounders
Bad: Only considering price, ignoring volume and volatility
Good: Include all relevant market conditions

### 3. Poor Overlap
Bad: Forcing matches when signal-days and non-signal-days are very different
Good: Only compare when propensity scores overlap reasonably

### 4. Overfitting the Propensity Model
Bad: Using 100 features to predict signal firing perfectly
Good: Use meaningful features that represent true confounders

---

## Key Takeaways

1. **Correlation ≠ Causation**: Just because your signal correlates with returns doesn't mean it causes them

2. **Confounders Are Sneaky**: Market conditions affect both your signal and returns

3. **Propensity Scores Create Fair Comparisons**: By matching similar situations, you can isolate the true effect

4. **This Works for Any Signal**: Technical indicators, fundamental factors, sentiment signals - all can be tested this way

5. **It's Not Magic**: You still need good data and careful analysis

---

## Summary Diagram

```
Traditional Analysis:
Signal → Returns (Is this real?)

With Propensity Score Analysis:

                    Market Conditions
                    /               \
                   ↓                 ↓
              Signal Fires       Returns
                   |                 |
                   └───── ? ─────────┘
                       (Find this!)

Step 1: Calculate P(Signal | Market Conditions) = Propensity Score
Step 2: Match signal-days with non-signal-days (same propensity)
Step 3: Compare returns → TRUE CAUSAL EFFECT
```

---

## Learn More

- **Easy Reading**: "The Book of Why" by Judea Pearl (explains causation simply)
- **Practice**: Try implementing the code examples in this chapter
- **Explore**: Look at how professional quant funds use causal inference

Remember: The goal isn't to predict returns better - it's to understand whether your strategies ACTUALLY work, or just got lucky!
