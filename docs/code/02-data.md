# Code B · The Data Layer (`data/`)

This is pipeline **step 1**: turn raw network traffic into clean, numeric,
time-windowed graphs — without leaking the future. Five files:

| File | Job |
|------|-----|
| `schema.py` | define one canonical "flow" + the feature name lists |
| `synthetic.py` | generate fake-but-realistic traffic (so it runs with no dataset) |
| `preprocess.py` | load real CIC-IDS2018 CSVs, clean flows, z-score normaliser |
| `features.py` | cut flows into time windows, compute per-node feature vectors |
| `splits.py` | leakage-safe chronological train/test split |

---

## B.1 `schema.py` — one flow to rule them all

Different data sources describe network traffic differently. We map them all onto
one shape: the **`FlowRecord`**. A **flow** is one bidirectional conversation
between two machines (e.g., "HOST-01 talked to SERVER-00 for 2 seconds, sending
10 packets, receiving 12").

```python
NODE_FEATURES = [
    "connection_frequency", "unique_destinations", "unique_ports",
    "failed_connections", "outbound_ratio", "mean_packet_rate",
    "mean_byte_rate", "mean_iat",
]
EDGE_FEATURES = ["packets", "bytes", "duration", "packet_rate", "byte_rate"]
```

- These two lists **fix the order** of the numbers in every feature vector. If a
  node's vector is `[5, 2, 1, 0, 1.0, 24.4, 2616.9, 9.8]`, index 0 is always
  `connection_frequency`, index 1 always `unique_destinations`, etc. Fixing the
  order is essential — the model learns "column 1 means unique destinations", so
  the columns must never move.

```python
@dataclass
class FlowRecord:
    ts: float                 # start time in seconds
    src: str                  # source machine id
    dst: str                  # destination machine id
    src_port: int
    dst_port: int
    protocol: str             # 'TCP' | 'UDP' | 'ICMP'
    duration: float           # seconds
    fwd_packets: int          # packets sent forward (src→dst)
    bwd_packets: int          # packets sent back (dst→src)
    fwd_bytes: int
    bwd_bytes: int
    label: str = "Benign"     # dataset label — used ONLY for evaluation
```

- `@dataclass` auto-creates the constructor, so `FlowRecord(ts=1.0, src="A", ...)`
  just works. Each line is one field with a **type hint**.
- `label` has a **default** (`= "Benign"`), so it's optional when creating a flow.
- **Crucial:** `label` (is this flow part of an attack?) is used *only to score
  how well we did afterwards* — it is **never** fed to the model. Feeding labels
  in would be cheating (leakage).

### Derived properties (computed on demand)

```python
    @property
    def packets(self):     return self.fwd_packets + self.bwd_packets
    @property
    def bytes(self):       return self.fwd_bytes + self.bwd_bytes
    @property
    def packet_rate(self): return self.packets / self.duration if self.duration > 0 else float(self.packets)
    @property
    def byte_rate(self):   return self.bytes / self.duration if self.duration > 0 else float(self.bytes)
    @property
    def failed(self):      return self.bwd_packets == 0
    @property
    def is_attack(self):   return self.label.strip().lower() not in ("benign", "normal", "")
```

- `@property` (met in Part A) lets us write `flow.packets` — it *looks* like
  stored data but is computed each time. We store the raw counts and derive the
  rest, so there's one source of truth.
- `packet_rate` / `byte_rate` — "per second" versions. The `if self.duration > 0
  else ...` guard avoids **divide-by-zero** (a flow with zero duration).
- `failed` — a connection that got **no reply** (`bwd_packets == 0`). This is the
  fingerprint of **port scanning**: an attacker knocks on many doors; most don't
  answer. That's why `failed_connections` is such a strong attack feature.
- `is_attack` — normalises the label text (lowercase, trimmed) and returns True
  for anything that isn't "benign"/"normal"/empty. Used only for evaluation.

---

## B.2 `synthetic.py` — realistic fake traffic

Real datasets like CIC-IDS2018 are gigabytes and need downloading. To let the
project run *anywhere* instantly, we generate believable traffic ourselves. The
key insight (learned the hard way during development): **benign traffic must be
predictable**, or the forecaster can't tell attacks from noise. So each host gets
a stable **profile** and only jitters slightly around it.

```python
class _HostProfile:
    def __init__(self, rng, servers):
        self.flows_per_window = rng.randint(4, 7)
        self.home_servers = rng.sample(servers, k=min(2, len(servers)))
        self.port = rng.choice([80, 443, 53])
        self.fwd_pkts = rng.randint(12, 24)
        ...
```

- A **profile** is a host's stable "personality": how many connections it makes
  per window, which 2 servers it usually talks to, its usual port and packet
  sizes. Set once, reused every window (with small noise). This makes normal
  behaviour learnable — a host reliably does ~5 connections to its 2 home
  servers.
- The leading underscore in `_HostProfile` marks it **internal** to this module.

```python
def generate_synthetic_flows(num_windows=40, window_seconds=60, num_hosts=18,
                             num_servers=4, attack_start_window=30,
                             attack_type="lateral_movement", seed=1337):
    rng = Rng(seed).spawn("synthetic")
    hosts = [_host_id(i) for i in range(num_hosts)]        # HOST-00, HOST-01, ...
    servers = [_server_id(i) for i in range(num_servers)]  # SERVER-00, ...
    profiles = {h: _HostProfile(rng, servers) for h in hosts}
    chain = rng.sample(hosts, k=min(3, num_hosts))   # the infection chain
    chain_starts = [0, 3, 6]                          # staggered activation
```

- Note `Rng(seed).spawn("synthetic")` — reproducible randomness from Part A.
- `hosts`/`servers` are built with **list comprehensions**; `profiles` with a
  **dict comprehension** (`{key: value for ...}`).
- `chain` = three hosts chosen to become compromised one after another; this is
  what lets us *demonstrate propagation* (an attack spreading A→B→C). Without a
  chain, only one host would ever look anomalous and there'd be nothing to
  "spread."

```python
    for w in range(num_windows):
        t0 = float(w * window_seconds)
        for h in hosts:
            _benign_host_window(flows, rng, h, profiles[h], t0, window_seconds)
        if w < attack_start_window:
            continue
        if attack_type == "lateral_movement":
            for idx, actor in enumerate(chain):
                start = attack_start_window + chain_starts[idx]
                if w < start:
                    continue
                forced = chain[idx + 1] if idx + 1 < len(chain) else None
                _inject_lateral_movement(flows, rng, actor, hosts, servers,
                                         scan_ports, t0, window_seconds,
                                         w - start, forced_target=forced)
        elif attack_type == "exfiltration":
            _inject_exfiltration(...)
```

Line by line:
- For each **window** `w`, first every host emits its normal benign traffic.
- `t0 = w * window_seconds` — the window's start time in seconds (window 30 →
  t0 = 1800). Anchoring to a clean grid matters for windowing later.
- Before `attack_start_window`, we `continue` (skip the attack code) — pure
  benign period the model trains on.
- For lateral movement, we walk the `chain`. Each actor "activates" at a
  staggered time (`chain_starts` = 0, 3, 6 windows after the attack begins), so
  the infection visibly spreads over time. `enumerate(chain)` gives both the
  index and the item.
- `forced = chain[idx+1]` forces each attacker to also contact the *next* chain
  member — guaranteeing the infection edge exists in the graph so the propagation
  engine can attribute the spread.

The two injectors just add flows with attack-shaped features:
- `_inject_lateral_movement` — the actor contacts *many* hosts on *many* ports,
  most getting **no reply** (`bwd_packets=0` → `failed`), then pivots to a server.
  This spikes `unique_destinations`, `unique_ports`, `failed_connections`,
  `connection_frequency`.
- `_inject_exfiltration` — the actor opens several *huge outbound* transfers to
  one server, spiking `mean_byte_rate` and `outbound_ratio`. It's deliberately
  subtler (ramps up), which is why the system detects it later and with lower
  recall — reported honestly.

**Takeaway:** the generator encodes *what attacks look like in feature space*.
Understanding it teaches you what signals a detector should hunt for.

---

## B.3 `preprocess.py` — real data, cleaning, and the z-score normaliser

### Loading CIC-IDS2018 CSVs

```python
_CIC_ALIASES = {
    "ts":       ["Timestamp", "timestamp", "flow_start"],
    "src":      ["Src IP", "Source IP", "src_ip", "Src"],
    "dst":      ["Dst IP", "Destination IP", "dst_ip", "Dst"],
    ...
}
```

- Public network-security CSVs are *inconsistent* — the same column might be
  "Src IP" in one file and "Source IP" in another. This **alias table** lists the
  acceptable spellings for each field, so our loader is robust to variations.

```python
def hash_asset(identifier, salt="sentinelx"):
    digest = hashlib.sha256(f"{salt}:{identifier}".encode()).hexdigest()
    return f"N{digest[:10]}"
```

- Turns a raw IP (e.g. `10.0.0.5`) into a stable, non-reversible node key like
  `Nab12cd34ef`. This honours the "never feed raw IPs as numbers" rule and adds a
  bit of privacy. Same IP → same key (so we can still track a machine across
  windows), but you can't get the IP back from the key.

```python
def load_cic_ids_csv(text_or_path, hash_ips=True, is_path=True):
    ...
    reader = csv.DictReader(io.StringIO(content))
    for i, row in enumerate(reader):
        raw_dur = _to_float(_pick(row, _CIC_ALIASES["duration"]), 0.0)
        duration = raw_dur / 1_000_000.0 if raw_dur > 1000 else raw_dur   # µs → s
        protocol = _PROTO_NAMES.get(str(proto_raw).strip(), ...)          # 6 → TCP
        records.append(FlowRecord(...))
    return records
```

- `csv.DictReader` (stdlib) reads each CSV row into a dictionary keyed by column
  name — no pandas needed.
- `_pick(row, aliases)` returns the first alias that exists in the row (handles
  the spelling variations).
- CIC durations are in **microseconds**; we convert to seconds (`/ 1_000_000`).
  Protocol numbers (6, 17) are mapped to names (TCP, UDP).
- IPs are hashed by default. Every row becomes a `FlowRecord`, so downstream code
  never knows whether the data was synthetic or real. That's the power of a
  **canonical schema**.

### Cleaning

```python
def clean_flows(records):
    cleaned = []
    for r in records:
        if r.duration < 0:                       continue
        if r.packets <= 0 and r.bytes <= 0:      continue   # empty flow, no signal
        if any(math.isinf(x) or math.isnan(x) for x in (r.ts, r.duration)): continue
        cleaned.append(r)
    cleaned.sort(key=lambda f: f.ts)
    return cleaned
```

- Real data is dirty: negative durations, empty flows, `NaN`/`Inf` values (from
  divide-by-zero in the capture tool). We drop those. `NaN` = "Not a Number",
  `Inf` = infinity — both poison maths downstream.
- `cleaned.sort(key=lambda f: f.ts)` — sort by timestamp so flows are in time
  order (essential for windowing). `key=lambda f: f.ts` means "sort by each
  flow's `ts`."

### The Normaliser (z-score) — with leakage protection built in

```python
@dataclass
class Normalizer:
    mode: str = "zscore"
    mins: list[float] = field(default_factory=list)
    maxs: list[float] = field(default_factory=list)
    means: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)
    _fitted: bool = False

    def fit(self, rows):
        ...
        self.means = [sum of each column / n]
        self.stds  = [sqrt(mean of (x-mean)^2), floored at 1e-6]
        self._fitted = True
        return self

    def transform_vector(self, vec):
        if not self._fitted:
            raise RuntimeError("Normalizer.transform called before fit (potential leakage)")
        if self.mode == "minmax":
            ... return clamped (v - min)/(max - min)
        return [(v - self.means[j]) / self.stds[j] for j, v in enumerate(vec)]
```

This small class is where the anti-leakage discipline lives:
- **`fit(rows)`** computes the mean and standard deviation of each feature — but
  you must pass it *training* rows only. It also records mins/maxs (for the
  optional min-max mode) and sets `_fitted = True`.
- **`transform_vector`** applies the z-score `(v - mean) / std`. The std is
  floored at `1e-6` so a constant feature (std = 0) doesn't cause divide-by-zero.
- **The guard:** if you call `transform` before `fit`, it *raises an error*
  mentioning leakage. This makes the classic mistake — normalising before
  splitting — impossible to do silently. That single `if not self._fitted:` line
  is a real engineering safeguard.
- `mode="zscore"` is the default (amplifies out-of-distribution attack spikes,
  as explained in concepts). `mode="minmax"` exists for cases where a bounded
  0–1 range is wanted.

---

## B.4 `features.py` — windowing and per-node features

### Cutting time into windows

```python
@dataclass
class WindowSlice:
    index: int
    start: float
    end: float
    flows: list[FlowRecord]

def build_windows(flows, window_seconds):
    ordered = sorted(flows, key=lambda f: f.ts)
    t_min = (ordered[0].ts // window_seconds) * window_seconds   # snap to grid
    t_max = ordered[-1].ts
    num_windows = int((t_max - t_min) // window_seconds) + 1
    buckets = [[] for _ in range(num_windows)]
    for f in ordered:
        idx = int((f.ts - t_min) // window_seconds)
        idx = min(idx, num_windows - 1)
        buckets[idx].append(f)
    ... return a WindowSlice per bucket (empty ones kept)
```

- A **`WindowSlice`** groups all flows that happened in one time window.
- `t_min = (ordered[0].ts // window_seconds) * window_seconds` — **grid
  anchoring.** `//` is integer division. This rounds the start time *down* to a
  clean multiple of the window size. Why it matters: the synthetic generator
  places window-30 flows at t = 1800; if we anchored to the first flow's actual
  time (~6s), everything would shift by one window and "attack starts at 30"
  wouldn't line up with graph window 30. Snapping to the grid keeps generator
  windows and graph windows perfectly aligned. (This exact off-by-one was a real
  bug we fixed — the tests now guard it.)
- `idx = int((f.ts - t_min) // window_seconds)` — which bucket a flow falls into.
- **Empty windows are kept** (as empty slices), so the time series has no silent
  gaps — important for a forecaster that assumes evenly spaced steps.

### Computing a node's feature vector

```python
def window_node_features(flows):
    outbound = {}       # node → its outgoing flows
    inbound_bytes = {}  # node → total bytes received
    all_nodes = set()
    for f in flows:
        all_nodes.add(f.src); all_nodes.add(f.dst)
        outbound.setdefault(f.src, []).append(f)
        inbound_bytes[f.dst] = inbound_bytes.get(f.dst, 0) + f.bytes

    features = {}
    for node in all_nodes:
        out_flows = outbound.get(node, [])
        connection_frequency = float(len(out_flows))
        unique_destinations  = float(len({f.dst for f in out_flows}))
        unique_ports         = float(len({f.dst_port for f in out_flows}))
        failed_connections   = float(sum(1 for f in out_flows if f.failed))
        out_bytes = sum(f.bytes for f in out_flows)
        in_bytes  = inbound_bytes.get(node, 0)
        outbound_ratio = out_bytes / (out_bytes + in_bytes) if (out_bytes+in_bytes) else 0.0
        mean_packet_rate = mean of f.packet_rate over out_flows (or 0)
        mean_byte_rate   = mean of f.byte_rate over out_flows (or 0)
        mean_iat         = mean gap between consecutive out_flows (or 0)
        features[node] = [connection_frequency, unique_destinations, unique_ports,
                          failed_connections, outbound_ratio, mean_packet_rate,
                          mean_byte_rate, mean_iat]
    return features
```

Every feature computed here maps to a concept from Part 2. Notes on the Python:
- `outbound.setdefault(f.src, []).append(f)` — "get the list for this source, or
  create an empty one, then append." A common dict idiom for grouping.
- `{f.dst for f in out_flows}` — a **set comprehension**. A **set** holds only
  *unique* items, so its length is the count of *distinct* destinations. Same
  trick for unique ports.
- `sum(1 for f in out_flows if f.failed)` — counts flows where `failed` is True.
- `mean_iat` — the **inter-arrival time**: sort the node's flows by time, take the
  gaps between consecutive ones, average them. A scanner fires many connections
  in a burst → tiny gaps → small IAT; a normal host is spaced out.
- The order of the returned list **exactly matches `NODE_FEATURES`** (there's even
  an `assert` in the real code checking the length). This is the contract the
  whole model relies on.

Note: features here are **raw** (not yet normalised). Normalisation happens later
in the pipeline, *after* the train/test split, using training stats only — that's
how leakage is avoided.

---

## B.5 `splits.py` — the leakage-safe split

```python
@dataclass
class TemporalSplit:
    train_indices: list[int]
    test_indices: list[int]

    @property
    def boundary(self):
        return self.test_indices[0] if self.test_indices else len(self.train_indices)
```

- A `TemporalSplit` just holds two lists of window indices: which windows are for
  training and which for testing. `boundary` is the first test window.

```python
def temporal_split(num_windows, test_fraction=0.3):
    n_test = max(1, round(num_windows * test_fraction))
    n_test = min(n_test, num_windows - 1) if num_windows > 1 else 0
    boundary = num_windows - n_test
    return TemporalSplit(train_indices=list(range(boundary)),
                         test_indices=list(range(boundary, num_windows)))
```

- The **last** `test_fraction` of windows (e.g. the last 30%) become the test
  set; everything before is training. Because windows are in time order, this is
  automatically **chronological**: we train on the past, test on the future.
- The `max(1, ...)` / `min(..., num_windows - 1)` clamps guarantee at least one
  window on each side.

```python
def assert_no_leakage(split):
    train, test = split.train_indices, split.test_indices
    if not train:                       raise AssertionError("empty training set")
    if set(train) & set(test):          raise AssertionError("train/test overlap")
    if test and max(train) >= min(test):raise AssertionError("time travel!")
    if train != sorted(train) or test != sorted(test):
                                        raise AssertionError("not chronological")
```

This is the **leakage tripwire**. It refuses to proceed unless:
- training isn't empty,
- train and test don't **overlap** (`&` is set intersection),
- no training window is at or after any test window (`max(train) >= min(test)`
  would be "training on the future" — time travel),
- both lists are in increasing order.

The pipeline calls this right after splitting. If anyone ever changes the split
logic and introduces a leak, this raises immediately and the tests fail. This is
what "trustworthy ML" looks like in code: a few lines that make cheating loud.

---

## Recap

Step 1 of the pipeline is complete: raw traffic (synthetic or real CSV) →
cleaned `FlowRecord`s → grouped into grid-aligned time windows → per-node feature
vectors → a strictly chronological, leak-checked train/test split. The z-score
`Normalizer` is ready to be fitted on training data only.

Next: [Code C — graph + models](03-graph.md)
