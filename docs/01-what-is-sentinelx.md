# 1 · What is Sentinel-X (the big idea)

No code in this part. Just intuition. Read it slowly.

## 1.1 The problem, told as a story

Imagine you are a security guard watching CCTV for a large office building.
Most security systems work like a **tripwire**: an alarm rings *after* someone
opens a door they shouldn't. By then, the intruder is already inside.

Computer networks have the same problem. Traditional tools ("intrusion
detection systems") fire an alert *after* something matches a known bad
pattern — a known virus signature, a blocked port, etc. Two big weaknesses:

1. **They are reactive.** The bad thing already happened.
2. **They miss slow, patient attacks.** A clever attacker moves quietly: log in
   here, look around there, copy a file next week. No single step looks alarming.
   It's the **trajectory** — the *sequence* of small moves — that is dangerous.

Think of a burglar casing a neighbourhood. One person walking down a street is
nothing. But the *same* person walking past every house, trying each gate,
returning at night — that **pattern over time** is the real signal.

## 1.2 The idea: forecast, don't just react

Sentinel-X flips the question. Instead of asking:

> "Did something bad *already* match a rule?"

it asks:

> "Based on how this network normally behaves, **what will it look like in the
> next few minutes** — and is reality drifting away from that prediction?"

This is **forecasting**: predicting the future from the past. Weather apps
forecast rain; Sentinel-X forecasts the *state of the network*.

The insight: **an attack is, by definition, the network doing something it
doesn't normally do.** If we can predict "normal", then anything the prediction
*fails* to see coming is suspicious — automatically, without needing a rule for
every possible attack.

## 1.3 Why a "graph"?

A network is not a flat list of events. It is a web of **who talks to whom**.

- Each computer (a "host") or server is a **node** (a dot).
- Each connection between two of them is an **edge** (a line).

That picture — dots connected by lines — is called a **graph**. (Nothing to do
with bar charts. In computer science, "graph" means "network of connected
things".)

```
   HOST-01 ───▶ SERVER-00
      │             ▲
      ▼             │
   HOST-02 ─────────┘
```

Why does this matter? Because attacks *spread through the connections*. A
compromised laptop attacks its neighbours; a neighbour attacks a server. If you
only look at each computer in isolation (a flat table), you miss the spread. If
you look at the **graph**, the spread is visible as a path lighting up.

## 1.4 Why "dynamic"?

The graph is not fixed. Every minute, different computers talk to different
others. So we take a **snapshot of the graph every time window** (say, every 60
seconds) and line them up:

```
 time →   G₁ → G₂ → G₃ → G₄ → G₅ ...
         (each Gₜ is one snapshot of the network)
```

A graph that changes over time like this is a **dynamic graph**. Sentinel-X
learns the *rules of motion*: given the last few snapshots, what does the next
snapshot look like? That learned "physics of the network" is what we call the
**world model**.

## 1.5 The four questions Sentinel-X answers

For any moment, it answers four things at once — and this is what makes it more
honest than a simple alarm:

1. **What happens next?** — the forecast (the next few snapshots).
2. **How sure are we?** — the **uncertainty**. A forecast with a big "±" is
   worth less than a confident one, and the system says so out loud.
3. **Have we ever seen behaviour like this?** — the **novelty** score. Brand-new
   behaviour (a never-before-seen attack) is flagged as "unknown", even if we
   can't name it.
4. **What if we intervened?** — the **counterfactual**. Before touching the real
   network, an analyst can ask "what if I isolate this laptop?" and see whether
   the predicted future gets safer.

## 1.6 A concrete walk-through (no code)

Here is the exact story the demo tells, in human terms:

1. The network hums along normally for a while. Sentinel-X watches and learns
   what "normal" looks like for each computer (how many others it talks to, how
   much data it sends, etc.).
2. Suddenly **HOST-15** starts contacting *many* other machines it never talked
   to, on unusual ports, and most of those attempts get no reply (it's
   **scanning** — knocking on many doors to see which open). This is classic
   **lateral movement** (an attacker spreading sideways through the network).
3. Sentinel-X's forecast for HOST-15 was "keep behaving normally". Reality
   wildly disagrees. The gap between prediction and reality — the **deviation
   score** — spikes. HOST-15 is flagged **anomalous**.
4. A minute later, a machine HOST-15 contacted, **HOST-06**, starts behaving
   the same way. The **propagation** layer notices the anomaly *spreading along
   an edge* from HOST-15 to HOST-06, like an infection. It even computes an
   "effective reproduction number" (borrowed from epidemiology — how fast a
   disease spreads).
5. The analyst opens the **counterfactual** screen and asks: *"What if I had
   isolated HOST-15 at the start?"* The system re-runs the forecast with HOST-15
   removed and shows the risk to the servers dropping from 77% to 0%. But if
   they ask the same thing *after* the infection already spread to three
   machines, isolating just one barely helps (risk stays high). That is a real,
   honest lesson: **early action matters.**
6. Finally, the **CyberChronicle** writes the whole episode in plain English,
   tagged with the industry-standard **MITRE ATT&CK** stage names ("Discovery",
   "Lateral Movement"), so a human can read it like a story and hand it off.

That is the entire product. Everything else in this book is *how* each of those
steps is actually computed.

## 1.7 Who uses it

| User | What they need it for |
|------|-----------------------|
| A security analyst (SOC) | Early warning that an attack is spreading, in plain language |
| A researcher | A sandbox to experiment with graph-based forecasting on network data |
| A student / hackathon judge | A demonstrable, honestly-built, novel system |

## 1.8 The guiding principle: *honest before impressive*

This is worth repeating because it shaped every design choice:

- We only claim a forecast horizon we actually validated.
- We keep a fancy metric only if experiments show it *helps*.
- We never treat auto-generated text as ground-truth security judgement.
- Every score is shown **with its uncertainty**, never as a bare number.

Restraint like this is itself a feature. It's the difference between a demo that
looks cool and a system an engineer can trust.

---

You now understand *what* the project does and *why*. Next we build up **every
concept** it uses, still without touching code.

Next: [Part 2 — Concepts from scratch](02-concepts.md)
