# How to Read the Decision Tree Visualization

## Overview
The Decision Tree shows **exactly how your trained PPO agent decides whether to KEEP or SWITCH a traffic light phase**.

---

## Step-by-Step Guide

### 1. **Start at the TOP Node**
```
┌─────────────────────────┐
│   Queue_Lane_4 <= 9.5   │  ← START HERE
│                         │
│  46% | 54%              │
│  Keep  Switch           │
└─────────────────────────┘
```

- **Feature Name**: `Queue_Lane_4` - the first thing the agent checks
- **Threshold**: `<= 9.5` - the condition being tested
- **Percentages**: `46% Keep, 54% Switch` - what % of traffic chose each action

---

### 2. **Follow the LEFT Branch (✓ condition is TRUE)**
If the feature value **≤ threshold**, go LEFT:

```
           Queue_Lane_4 <= 9.5
                 /
                /  (YES, ≤ 9.5)
               /
    Queue_Lane_1 <= 3.0
    (Next decision)
```

**Interpretation**: "IF the queue at Lane 4 is small (≤ 9.5)..."

---

### 3. **Follow the RIGHT Branch (✗ condition is FALSE)**
If the feature value **> threshold**, go RIGHT:

```
           Queue_Lane_4 <= 9.5
                    \
                     \ (NO, > 9.5)
                      \
    Traffic_Light_Phase <= 0.5
    (Next decision)
```

**Interpretation**: "ELSE if the queue at Lane 4 is large (> 9.5)..."

---

### 4. **Keep Following Down Until You Reach a LEAF**

```
Queue_Lane_4 <= 9.5 (YES)
    ↓
Queue_Lane_1 <= 3.0 (YES)
    ↓
Emergency_Lane_3 <= 1.0 (YES)
    ↓
┌──────────────────────┐
│  SWITCH PHASE        │  ← FINAL DECISION
│  78% | 22%           │
│  Switch  Keep        │
└──────────────────────┘
```

---

## Understanding Node Colors

| Color | Meaning | What it Means |
|-------|---------|---------|
| 🔵 **Blue** | Mostly "KEEP" | The agent usually keeps the phase in this situation |
| 🟠 **Orange/Red** | Mostly "SWITCH" | The agent usually switches the phase in this situation |
| 🟡 **Yellow** | Mixed 50/50 | The agent is uncertain - about equal chance |

---

## Interpreting the Percentages

In each node, you see two numbers:

```
      43% | 57%
     Keep  Switch
```

- **43%**: In this situation, 43% of observed cases the agent chose "KEEP PHASE"
- **57%**: In this situation, 57% of observed cases the agent chose "SWITCH PHASE"

**The color shows the majority**: Since 57% > 43%, the node will be orange (SWITCH).

---

## A Complete Example

Let's say we follow this path:

```
┌─────────────────────┐
│ Queue_Lane_4 <= 9.5 │ (Is queue small?)
│   46% | 54%         │
└────────┬────────────┘
         │
    YES (≤ 9.5)
         │
         ↓
┌─────────────────────┐
│ Queue_Lane_1 <= 3.0 │ (Is Lane 1 almost free?)
│   62% | 38%         │
└────────┬────────────┘
         │
    YES (≤ 3.0)
         │
         ↓
┌─────────────────────┐
│ KEEP PHASE          │  ← FINAL DECISION
│   72% | 28%         │
└─────────────────────┘
```

**English Translation**:
> "If Lane 4's queue is small (≤ 9.5) AND Lane 1 is almost empty (≤ 3.0), then KEEP THE CURRENT PHASE because it's working well (72% chose this)."

---

## Understanding Top 5 Factors

From the interpretation file:

```
1. Queue_Lane_4        - Importance: 18.9%  ← MOST IMPORTANT
2. Queue_Lane_1        - Importance: 16.3%
3. Traffic_Light_Phase - Importance: 15.4%
4. Emergency_Lane_3    - Importance: 15.1%
5. Bus_Lane_11         - Importance: 9.8%
```

**What it means**:
- `Queue_Lane_4` has the **highest influence** on switching decisions (18.9%)
- `Queue_Lane_1` is the **2nd most important** (16.3%)
- Other lanes matter less
- The agent primarily looks at **queues and emergency presence**

---

## Questions You Can Answer

### Q: "When does the agent switch the phase?"
**A**: Follow the tree from top. Nodes colored orange (>50% Switch) indicate switching scenarios.

### Q: "Why did it switch here but not there?"
**A**: Look at the path. Different conditions (queue sizes, emergency vehicles) lead to different decisions.

### Q: "Which sensor matters most?"
**A**: Check the top 5 factors. `Queue_Lane_4` is by far the most important.

### Q: "Is the tree confident in its decisions?"
**A**: Look at percentages. 90% vs 10% = confident. 55% vs 45% = uncertain.

---

## Key Insights

✅ **The agent is looking at QUEUE LENGTHS primarily** (Lane 4, Lane 1, etc.)
✅ **Emergency vehicle presence matters** (Emergency_Lane_3 is in top 5)
✅ **Traffic light phase helps** (16.3% importance - not the main factor)
✅ **Bus vehicles matter less** (Bus_Lane_11 only 9.8%)

---

## Tips for Better Understanding

1. **Print it out**: The PNG is large. Print it or view it full-screen.
2. **Use a highlighter**: Trace one path from top to bottom to understand one scenario.
3. **Start with extreme values**: 
   - Queue_Lane_4 = 0 (very empty) → Which path? → What decision?
   - Queue_Lane_4 = 20 (very full) → Which path? → What decision?
4. **Compare branches**: Why does LEFT branch (small queue) choose differently than RIGHT (large queue)?

---

## Still confused?

The interpretation file (`decision_tree_logic_interpretation.txt`) has:
- ✅ Top 5 most important features
- ✅ How to read the visual tree
- ✅ Example scenarios
- ✅ Key insights

Check that file for more details!
