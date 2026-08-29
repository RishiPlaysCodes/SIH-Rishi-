# 2 · Every Concept From Scratch

This is the longest part on purpose. If you understand everything here, the code
will feel obvious. Each concept has: a plain definition, an analogy, and usually
a tiny worked example with real numbers.

Take breaks. You don't need to memorise — just understand.

---

## 2.1 Graphs (nodes and edges)

A **graph** is a set of *things* and the *connections* between them.

- A **node** (also called a "vertex") is a thing. Here: a computer or server.
- An **edge** is a connection between two nodes. Here: "these two machines
  talked to each other".

Analogy: a graph is a map of friendships. People are nodes; "is friends with"
is an edge.

A **directed** graph means edges have a direction (an arrow). "HOST-01 sent data
*to* SERVER-00" is different from the reverse. We use directed edges because in
security, *who initiated* the connection matters.

**Adjacency** just means "who is directly connected to whom". The list of a
node's outgoing neighbours is its adjacency list. Example:

```
edges: A→B, A→C, B→C
adjacency:  A: [B, C]     B: [C]     C: []
```

That's the whole idea. A graph is dots and arrows.

---

## 2.2 Dynamic graphs (graphs over time)

A **dynamic graph** is a graph that changes over time. We capture it as a
*sequence of snapshots*, one per **time window**.

A **time window** is just a fixed slice of time — e.g., "every 60 seconds". All
the connections that happened in that minute become one snapshot.

```
window 1 (0–60s):   A→B, A→C
window 2 (60–120s): A→B, B→C, C→D
window 3 ...
```

We write the snapshot at time *t* as **Gₜ = (Vₜ, Eₜ, Xₜ)**:
- *Vₜ* = the nodes present in that window,
- *Eₜ* = the edges,
- *Xₜ* = the **features** of the nodes (explained next).

Don't be scared of the notation — it's just "the graph at time t, with its dots,
lines, and the numbers describing each dot."

---

## 2.3 Features and feature vectors (turning behaviour into numbers)

Computers don't understand "this host is acting suspicious". They understand
**numbers**. So for each node we compute a list of numbers that describe its
behaviour in that window. That list is a **feature vector**.

A **feature** is one measurable property. For a host we use 8 features:

| Feature | Plain meaning |
|---|---|
| `connection_frequency` | how many outgoing connections it made |
| `unique_destinations` | how many *different* machines it contacted |
| `unique_ports` | how many *different* ports (services) it hit |
| `failed_connections` | how many attempts got no reply |
| `outbound_ratio` | fraction of its traffic that was outgoing vs incoming |
| `mean_packet_rate` | average packets per second |
| `mean_byte_rate` | average bytes per second |
| `mean_iat` | mean "inter-arrival time" = average gap between its connections |

So a host might become the vector `[5, 2, 1, 0, 1.0, 24.4, 2616.9, 9.8]`.

Why these? Because attacks show up in them. A scanning attacker's
`unique_destinations`, `unique_ports`, and `failed_connections` all shoot up.
A data thief's `mean_byte_rate` and `outbound_ratio` shoot up. The features are
chosen so that *bad behaviour changes the numbers*.

> **Key rule:** we never feed a raw IP address (like `10.0.0.5`) into the maths
> as a number — that would be meaningless (is `10.0.0.5` "less than" `10.0.0.6`?
> No). IP addresses are only used as node *identities* (labels), or hashed. Only
> genuine behavioural quantities become features.

---

## 2.4 Normalisation (putting features on a fair scale)

Problem: `mean_byte_rate` might be in the thousands, while `unique_ports` is a
small number like 2. If we compare them directly, the big-numbered feature drowns
out the small one. We must put every feature on a **comparable scale**. That is
**normalisation**.

Two common ways:

### Min–max scaling
Squeeze every value into the range 0 to 1:
```
scaled = (value − min) / (max − min)
```
If a feature ranged 0–10 during training, then 5 becomes 0.5, 10 becomes 1.0.

**The trap (this is crucial and it bit us during development):** min–max
*clamps*. If the biggest value we ever saw in normal traffic was 2 different
destinations, then `max = 2`. Now an attacker contacts 20 destinations. Min–max
maps *both* 2 and 20 to `1.0` — because anything ≥ max becomes 1.0. **The attack
signal is erased!** The very spike we needed to see gets flattened.

### Z-score standardisation (what we use)
Measure each value in "how many standard deviations from the average":
```
z = (value − mean) / standard_deviation
```
- **mean** = the average value during normal training.
- **standard deviation (std)** = the typical spread around that average (a small
  std means values are usually close to the mean).

Worked example. Suppose during normal traffic `unique_destinations` was almost
always 2, with mean = 2 and std = 0.2. Now:
- A normal host with 2 destinations → z = (2−2)/0.2 = **0** (perfectly normal).
- An attacker with 12 destinations → z = (12−2)/0.2 = **50** (screamingly
  abnormal).

Z-score does **not** clamp. The further out-of-distribution a value is, the
bigger the z. That's exactly what an anomaly detector wants: it *amplifies* the
weird instead of hiding it. This single choice is why the system detects attacks
well.

---

## 2.5 What is a "model"? What is "training"?

A **model** is a function with adjustable knobs that learns a pattern from data.

- **Training** (or "fitting") = tuning the knobs so the model matches examples.
- **Prediction** (or "inference") = using the tuned model on new input.

Analogy: learning to catch a ball. After watching many throws (training), your
brain builds an internal model of "how balls fly", so you can predict where the
next one lands (prediction).

In Sentinel-X the model is a **world model**: it learns "given the network's
state now, what will it be next window?"

We build up model complexity honestly, simplest first:

1. **Persistence** — "the next window looks exactly like this one." No learning
   at all. A dumb but useful baseline; any real model must beat it.
2. **EWMA** (Exponentially Weighted Moving Average) — "the next window looks like
   a *smoothed average* of recent windows, weighting recent ones more."
3. **Linear transition (ridge regression)** — actually *learns* how features
   evolve. This is our main model. Explained next.
4. (Swap-in) **Graph Neural Network** — the heavy-duty version used in industry;
   we explain it in [Part 3](03-tech-stack.md).

Why baselines matter: if a fancy model can't beat "just predict no change", the
fancy model is worthless. Comparing against baselines is called an **ablation**
and it's a mark of honest engineering.

---

## 2.6 Linear regression → Ridge regression

### Linear regression (the idea)
Find the straight-line relationship between inputs and an output. "As X goes up,
Y goes up by this much." The classic `y = m·x + c`. Training finds the best `m`
(slope) and `c` (intercept) so the line fits the data points with least error.

With many inputs it becomes `y = w₁x₁ + w₂x₂ + ... + b`, where the `w`s are
**weights** (one knob per input) and `b` is the **bias** (the intercept). The
whole set of weights is often written as a matrix **W**.

In our world model, the input X is a node's feature vector *now*, and the output
is its predicted feature vector *next window*. So the model learns a matrix W
such that: `features_next ≈ features_now × W`.

### The "normal equation" (how we solve it exactly)
For linear models there's a famous closed-form solution (no trial-and-error
needed):
```
W = (XᵀX)⁻¹ Xᵀ Y
```
Don't panic. In words: gather all the (input, next-output) pairs into big tables
X and Y, then this formula spits out the best weights directly. `Xᵀ` means
"transpose" (flip rows and columns); `⁻¹` means "matrix inverse" (the matrix
version of division). We implement all of this by hand in `linalg.py` — you'll
see it's just loops.

### Ridge regression (a safer version)
Plain linear regression can become unstable — the weights blow up if inputs are
correlated or scarce. **Ridge regression** adds a small penalty that keeps the
weights modest:
```
W = (XᵀX + λI)⁻¹ Xᵀ Y
```
The only new thing is `+ λI` (lambda times the identity matrix). `λ` (lambda) is
a small number (we use 0.05) that says "prefer smaller, safer weights". `I` is
the **identity matrix** (1s on the diagonal, 0s elsewhere — the matrix version of
the number 1). This trick is called **regularisation**: nudging a model toward
simpler solutions so it generalises better. That's the whole difference.

---

## 2.7 Forecasting and K-step rollout

**Forecasting** = predicting future values from past ones.

A **one-step** forecast predicts the very next window. But we often want to look
further: **K-step** forecasting predicts K windows ahead (K = 1, 2, 3, ...).

How do we predict 3 steps ahead with a one-step model? We **roll forward**:
predict step 1, then *feed that prediction back in* to predict step 2, then again
for step 3. This is called **autoregressive rollout**.

```
now ─▶ predict T+1 ─▶ (pretend T+1 is real) ─▶ predict T+2 ─▶ ... ─▶ T+K
```

Important honesty point: the further you roll, the shakier it gets (errors
compound). So we *increase the uncertainty* with each step and never claim a
horizon we didn't validate.

---

## 2.8 Behavioural deviation (the core anomaly signal)

Here's where forecasting becomes detection.

For each window we do: **predict** the graph, then look at what **actually**
happened, and measure the **gap**. That gap is the **deviation score**
(written `Dₜ = d(Gₜ, Ĝₜ)`, read "the distance between the real graph Gₜ and the
predicted graph Ĝₜ"). Big gap = the network did something we didn't see coming =
suspicious.

The gap is broken into interpretable pieces:
- **feature error** — how wrong were the predicted node numbers?
- **node-state error** — overall per-node difference.
- **structural error** — did the *connections* change unexpectedly? (measured
  with the **Jaccard distance** — see below).
- **edge-state error** — did traffic volumes on edges change?
- **temporal error** — did the *amount of change* differ from what we expected?

These are combined with weights into one score between 0 and 1.

### Jaccard distance (comparing two sets of connections)
To compare "the set of edges I predicted" vs "the set that actually happened":
```
Jaccard similarity = (edges in both) / (edges in either)
Jaccard distance    = 1 − similarity
```
If prediction and reality share all edges → distance 0. Share none → distance 1.
Simple set overlap.

### Emphasising the worst feature
An attack often spikes just *one or two* features hugely while others stay
normal. If we simply *average* the error across all 8 features, one giant spike
gets diluted by seven calm ones. So the feature error is a **blend of the average
error and the single worst feature's error** — this keeps the signal sharp.

### Saturating (keeping scores in 0–1)
Because z-scores are unbounded (an attack can be 50 std out), the raw errors can
be huge. We squash them into 0–1 with a **saturating function**:
```
saturate(x) = x / (x + c)
```
As x grows, this creeps toward 1 but never exceeds it. `c` sets "how big counts
as fully anomalous". This gives a bounded, human-readable score.

Finally we threshold the score into three states: **normal**, **deviating**
(a bit off), **anomalous** (clearly wrong).

---

## 2.9 Uncertainty via MC-Dropout

A forecast without a confidence level is dangerous. **Uncertainty** tells you how
much to trust it.

**Dropout** is a technique where, on each run, we randomly "drop" (zero out) some
of the model's inputs. Do this many times and the predictions vary a little each
run. **MC-Dropout** ("Monte Carlo Dropout") means: run the model *N* times with
different random drops and look at the spread of answers.

- If all N runs agree → low spread → **confident** forecast.
- If the N runs disagree a lot → high spread → **uncertain** forecast.

"Monte Carlo" just means "estimate something by running many random trials"
(named after the casino). The spread is measured by the **standard deviation** of
the N predictions, and we label it LOW / MEDIUM / HIGH. Uncertainty naturally
grows the further ahead we forecast — as it should.

---

## 2.10 Propagation & the effective reproduction number

Attacks spread. The **propagation** layer treats an anomaly like an **infection
moving across the graph**: if node A is anomalous this window and it has an edge
to node B, and B *becomes* anomalous next window, we say the anomaly *propagated*
A → B.

From this we borrow three ideas from **epidemiology** (the science of how
diseases spread):

- **Propagation velocity** — how many new nodes get "infected" per second.
- **Propagation intensity** — how strong the abnormal behaviour is across the
  spread.
- **Effective reproduction number (Rₑ)** — the average number of *new*
  infections caused by the *currently* infected. This is literally the "R number"
  you heard about during COVID. Rₑ > 1 means the outbreak is growing; Rₑ < 1
  means it's dying out.

This turns a pile of individual alerts into a *story of spread* the analyst can
follow along a path: `HOST-15 → HOST-06 → HOST-07`.

---

## 2.11 Novelty / Out-of-Distribution (OOD)

Sometimes behaviour is weird in a way we've *never seen before* — not just a
known attack, but something genuinely new. Detecting that is **novelty
detection** or **out-of-distribution (OOD) detection** ("out of distribution" =
outside the range of what we trained on).

To do this we need to compare whole graphs. We compute an **embedding** for each
graph — a fixed-length list of numbers summarising it (its average node features
plus a few structural stats like node count and density). An embedding is like a
*fingerprint*: two similar graphs have similar embeddings.

Then novelty = a blend of:
- **embedding distance** — how far is this graph's fingerprint from anything seen
  in training? (measured with **Euclidean distance** — ordinary straight-line
  distance, the `√((a₁−b₁)² + (a₂−b₂)² + ...)` you learned in school, just with
  more dimensions).
- **prediction error** — how badly did the model forecast this window?
- **uncertainty** — how unsure was it?

The result is labelled on a scale: **KNOWN → FAMILIAR → UNUSUAL → EMERGING →
UNKNOWN**. "UNKNOWN" is the system honestly saying *"I've never seen anything
like this."*

---

## 2.12 Forecast stability

Can we trust a single forecast? One test: **perturb** it. **Perturbation** means
"nudge the input a tiny bit at random." We take the latest evidence, jiggle the
numbers slightly, and re-run the forecast several times.

- If the forecast barely moves → **STABLE** (robust, trustworthy).
- If a tiny nudge sends the forecast swinging wildly → **UNSTABLE** (fragile,
  treat with caution).

It's the software version of testing whether a table wobbles by pushing it
gently.

---

## 2.13 Counterfactuals ("what-if" simulations)

A **counterfactual** is a "what would have happened if...?" question. Here it
lets an analyst simulate a defensive action *before* doing it for real.

The available actions (interventions):
- `ISOLATE_NODE` — cut a machine off entirely.
- `BLOCK_EDGE` — cut one specific connection.
- `BLOCK_PORT` — block a service/port.
- `DISABLE_COMMUNICATION` — stop a node from talking.
- `RATE_LIMIT` — throttle a node's traffic.

The engine works like this:
```
Gₜ (now)                    → world model → Forecast A → Risk_A
Gₜ + intervention (Gₜ')     → world model → Forecast B → Risk_B
ΔRisk = Risk_A − Risk_B      (how much safer did we make the future?)
```

**Risk** here is computed structurally: from the compromised nodes, how many
servers ("crown jewels") can be *reached* by following edges? Isolating the right
node early cuts those paths and drops the risk. Isolating too late (after the
infection spread to many nodes) barely helps — and the tool shows that honestly.

---

## 2.14 MITRE ATT&CK mapping

**MITRE ATT&CK** is a free, industry-standard catalogue of attacker behaviours,
organised into stages of an intrusion: *Reconnaissance → Initial Access →
Execution → Persistence → Privilege Escalation → Discovery → Lateral Movement →
Collection → Exfiltration → Command and Control.*

Sentinel-X maps its findings onto these stage names *after the fact*, purely so a
human analyst gets familiar vocabulary ("ah, this is Lateral Movement"). It is an
**interpretation layer**, not a detector — it never decides anything on its own.

---

## 2.15 Data leakage (the #1 way ML projects secretly cheat)

**Data leakage** is when information from the "test" (future/unseen) data
secretly influences training. It makes a model look brilliant in the lab and fail
in reality. For a *time-based* problem like ours, there are two classic leaks:

1. **Shuffling time.** If you randomly shuffle windows into train/test, the model
   effectively "sees the future" during training. The only correct split is
   **chronological**: train on the past, test on the strictly-later future. This
   is a **temporal split**.
2. **Fitting the scaler on all data.** Remember z-score needs a mean and std. If
   you compute those over the *entire* dataset (including test), the test
   statistics leak into training. You must compute mean/std on **training data
   only**, then apply them to test.

Sentinel-X makes both leaks *hard to do by accident*: the split is enforced and
checked, and the normaliser refuses to transform before it's been fitted. There
are dedicated tests that fail loudly if leakage sneaks in. Preventing leakage is
boring and unglamorous — and it's exactly what separates trustworthy ML from
lab-toys.

---

## 2.16 Reproducibility, seeds, and configs

**Reproducibility** = anyone can re-run your experiment and get the *exact* same
result. This is a cornerstone of real science and real engineering.

Two enablers:
- A **seed.** Anything "random" in a computer is actually **pseudo-random**: a
  formula that produces a fixed sequence from a starting number called the
  **seed**. Same seed → same "random" sequence → same results every time.
- A **config file.** All the settings (how many hosts, which model, thresholds)
  live in one YAML file, not scattered in code. So a run is fully described by
  *config + seed*, and it's captured in the database for every experiment.

(**YAML** is a simple, human-friendly text format for settings — indentation and
`key: value` lines. We even wrote a tiny YAML reader ourselves so the project
needs zero libraries.)

---

## You made it 🎉

That is every concept in the project. If some felt fuzzy, that's normal — they'll
click when you see them *in code*. Keep this page open as a reference.

Next: [Part 3 — The tech stack, and why](03-tech-stack.md)
