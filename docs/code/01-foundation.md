# Code A · Foundation (`__init__.py`, `config.py`, `seeding.py`, `linalg.py`)

These four files are used by *everything else*. Master them and the rest is easy.

> How to read a code walkthrough: I show a chunk of real code, then explain it
> line by line underneath. New Python syntax is explained the first time.

---

## A.1 `sentinelx/__init__.py` — what makes the folder a package

```python
"""Sentinel-X: Predictive Network Behavior & Threat Forecasting Platform.
   ... (a docstring describing the package layout) ...
"""
__version__ = "0.1.0"
```

- The triple-quoted string at the top is a **docstring** — documentation Python
  stores on the module. Tools and `help()` can read it.
- `__version__` is a convention: a variable holding the package's version. The
  presence of this file (even nearly empty) is what tells Python "`sentinelx` is
  an importable package." That's the *only* required job of `__init__.py`.

---

## A.2 `sentinelx/seeding.py` — reproducible randomness

Recall from concepts: computers make **pseudo-random** numbers from a **seed**.
Same seed → same sequence → reproducible experiments. This file wraps that.

```python
from __future__ import annotations

import hashlib
import random


class Rng:
    def __init__(self, seed: int):
        self.seed = int(seed)
        self._r = random.Random(self.seed)
```

- `from __future__ import annotations` — a line you'll see atop most files. It
  makes Python treat type hints as plain text (not evaluate them at runtime).
  Two benefits: hints can reference types defined later, and there's zero runtime
  cost. Harmless to you as a reader; just know it's a best-practice header.
- `import hashlib` — stdlib hashing (we use SHA-256 below).
- `import random` — stdlib pseudo-random generator.
- `class Rng:` — we define our own small class instead of using the global
  `random` functions. Why? Because if two parts of the program both used the
  *global* random state, they'd interfere with each other's sequences. A
  dedicated object keeps each stream independent and predictable.
- `def __init__(self, seed: int):` — the **constructor**, run when you write
  `Rng(1337)`. `self` is the object being built; `seed: int` is a **type hint**
  saying "seed should be an int" (hints are documentation, not enforced).
- `self._r = random.Random(self.seed)` — create a private random generator
  seeded with our seed. The leading underscore in `_r` is a convention meaning
  "internal, don't touch from outside."

```python
    def uniform(self, lo=0.0, hi=1.0): return self._r.uniform(lo, hi)
    def gauss(self, mu=0.0, sigma=1.0): return self._r.gauss(mu, sigma)
    def randint(self, lo, hi):          return self._r.randint(lo, hi)
    def random(self):                   return self._r.random()
    def choice(self, seq):              return self._r.choice(seq)
    def shuffle(self, seq):             self._r.shuffle(seq)
    def sample(self, population, k):    return self._r.sample(population, k)
```

These are thin pass-throughs to Python's random generator — each is one useful
kind of randomness:
- `uniform(lo, hi)` — a random *decimal* between lo and hi (evenly likely).
- `gauss(mu, sigma)` — a random number from a **bell curve** (Gaussian/normal)
  centred at `mu` with spread `sigma`. Used to add realistic jitter.
- `randint(lo, hi)` — a random *whole number* between lo and hi (inclusive).
- `random()` — a decimal in [0, 1). Handy for "do X with probability p".
- `choice(seq)` — pick one item from a list.
- `shuffle(seq)` — randomly reorder a list *in place*.
- `sample(population, k)` — pick `k` *distinct* items.

```python
    def spawn(self, label: str) -> Rng:
        digest = hashlib.sha256(f"{self.seed}:{label}".encode()).hexdigest()
        return Rng(int(digest[:16], 16))
```

This is the clever bit. `spawn` creates a **child** random stream derived
deterministically from a text label. Line by line:
- `f"{self.seed}:{label}"` — an **f-string** (formatted string). If seed=1337 and
  label="mc-3", this becomes `"1337:mc-3"`.
- `.encode()` — turn the text into bytes (hashing works on bytes).
- `hashlib.sha256(...).hexdigest()` — compute the **SHA-256 hash**, a fixed-length
  scramble of the input, returned as a hex string. Same input → same hash,
  always.
- `digest[:16]` — take the first 16 hex characters.
- `int(..., 16)` — read those hex characters as a number (base 16).
- `return Rng(that_number)` — build a new `Rng` seeded with it.

Why bother? So different components ask for `rng.spawn("synthetic")`,
`rng.spawn("mc-3")`, etc., and each gets its *own* independent-but-reproducible
stream from the *one* master seed. Reproducibility without collisions.

```python
def make_rng(seed: int) -> Rng:
    return Rng(seed)
```

A tiny convenience function — some callers prefer `make_rng(1337)` over
`Rng(1337)`. Same thing.

---

## A.3 `sentinelx/linalg.py` — hand-written linear algebra

This is where we *build NumPy's core, ourselves*, so you see how it works. Data
types first:

```python
Vector = list[float]
Matrix = list[list[float]]
```

- These are **type aliases** — nicknames. A `Vector` is just "a list of floats"
  (e.g. `[1.0, 2.0]`). A `Matrix` is "a list of lists of floats" — a table of
  numbers, e.g. `[[1.0, 2.0], [3.0, 4.0]]` is a 2×2 matrix. `list[float]` is
  modern Python syntax for "list containing floats."

### Vector operations

```python
def vadd(a, b):    _check_len(a, b); return [x + y for x, y in zip(a, b)]
def vsub(a, b):    _check_len(a, b); return [x - y for x, y in zip(a, b)]
def vscale(a, s):  return [x * s for x in a]
def vdot(a, b):    _check_len(a, b); return sum(x * y for x, y in zip(a, b))
```

- `zip(a, b)` — pairs up elements: `zip([1,2],[3,4])` gives `(1,3),(2,4)`.
- `[x + y for x, y in zip(a, b)]` — a **list comprehension**: "build a new list
  by adding each pair." So `vadd` adds two vectors element-wise.
- `vscale` multiplies every element by a scalar `s`.
- `vdot` is the **dot product**: multiply matching elements and sum them.
  `vdot([1,2,3],[4,5,6]) = 1·4 + 2·5 + 3·6 = 32`. The dot product is the single
  most important operation in this file — matrix multiply is just many dot
  products.
- `_check_len(a, b)` (defined at the bottom) raises a clear error if the two
  vectors have different lengths — catching bugs early.

```python
def l2(a):          return math.sqrt(sum(x * x for x in a))
def euclidean(a, b): return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
def mse(a, b):       return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)
def cosine(a, b):    ... return vdot(a, b) / (l2(a) * l2(b))
```

- `l2` — the **length** (magnitude) of a vector: √(sum of squares). Named "L2"
  after the L2 norm.
- `euclidean` — the **straight-line distance** between two vectors (Pythagoras in
  many dimensions). Used constantly in novelty/stability.
- `mse` — **Mean Squared Error**: average of squared differences. The standard
  "how wrong is a prediction" measure. `** 2` means "squared".
- `cosine` — **cosine similarity**: measures the *angle* between two vectors
  (1 = same direction, 0 = perpendicular). Guards against divide-by-zero when a
  vector has zero length.

```python
def mean_vector(rows): ...   # average of a list of vectors, element-wise
def std_vector(rows):  ...   # standard deviation, element-wise
```

- `mean_vector` — given many feature vectors, returns their per-feature average.
- `std_vector` — per-feature **standard deviation** (spread). These two are
  exactly what the z-score normaliser needs (mean and std per feature), and what
  MC-Dropout uses to summarise the spread of many predictions.

### Matrix operations

```python
def transpose(m):  return [list(col) for col in zip(*m)]
```

- **Transpose** flips rows and columns: `[[1,2],[3,4]]` → `[[1,3],[2,4]]`.
- `zip(*m)` — the `*` unpacks the rows as separate arguments to `zip`, which then
  pairs the first element of each row, the second of each, etc. — exactly a
  transpose. A classic Python trick.

```python
def matmul(a, b):
    if len(a[0]) != len(b):
        raise ValueError(...)
    bt = transpose(b)
    return [[vdot(row, col) for col in bt] for row in a]
```

- **Matrix multiplication.** The rule: to multiply A (size *r×k*) by B (*k×c*),
  the inner sizes must match (`len(a[0]) == len(b)`), giving an *r×c* result.
- We transpose B so its columns become rows (`bt`), then every output entry is a
  **dot product** of a row of A with a column of B. The nested comprehension
  reads: "for each row in A, for each column in B, take their dot product."
- This one function is the workhorse behind training the model.

```python
def matvec(m, v):  return [vdot(row, v) for row in m]
```

- Matrix × vector: dot each row of the matrix with the vector. Used to *apply*
  the trained model to one node's features.

### Solving linear systems: `solve(A, B)`

This solves the equation **A · X = B** for the unknown X — the matrix version of
"divide B by A". We use it to compute the ridge-regression weights.

```python
def solve(a, b):
    n = len(a)
    ...
    aug = [list(a[i]) + list(b[i]) for i in range(n)]   # [A | B] side by side
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            aug[pivot][col] += 1e-12                     # nudge if ~zero
        aug[col], aug[pivot] = aug[pivot], aug[col]      # swap pivot row up
        pivot_val = aug[col][col]
        aug[col] = [v / pivot_val for v in aug[col]]     # normalise pivot row
        for r in range(n):
            if r == col: continue
            factor = aug[r][col]
            if factor != 0.0:
                aug[r] = [rv - factor * cv for rv, cv in zip(aug[r], aug[col])]
    return [row[n:n + m] for row in aug]                 # the X part
```

This is **Gauss–Jordan elimination** — the systematic version of the
"eliminate one variable at a time" method you may have done by hand in algebra.
The intuition, no need to memorise:
- Glue A and B together into one wide table `aug` (called the **augmented
  matrix**).
- For each column, pick the row with the biggest value in that column
  (**partial pivoting** — improves numerical accuracy), move it up, divide it so
  its pivot becomes 1, then subtract multiples of it from all other rows to zero
  out that column everywhere else.
- After doing this for every column, the left half has become the identity
  matrix and the right half *is* the answer X. We slice it out with
  `row[n:n+m]`.
- `1e-12` means 0.000000000001 — a tiny number. The nudge avoids dividing by
  (near) zero when a matrix is **singular** (has no clean inverse), keeping the
  solver stable instead of crashing.

You do not need to reproduce this from memory — just know: *`solve` finds the X
that satisfies A·X = B, robustly.*

### Ridge regression: `ridge_fit(x, y, lam)`

This is the training maths for our world model. Recall the formula from concepts:
`W = (XᵀX + λI)⁻¹ Xᵀ Y`.

```python
def ridge_fit(x, y, lam):
    xt = transpose(x)                 # Xᵀ
    xtx = matmul(xt, x)               # XᵀX
    in_dim = len(xtx)
    reg = [[xtx[i][j] + (lam if i == j else 0.0) for j in range(in_dim)]
           for i in range(in_dim)]    # XᵀX + λI  (add λ only on the diagonal)
    xty = matmul(xt, y)               # Xᵀ Y
    return solve(reg, xty)            # solve (XᵀX + λI) · W = Xᵀ Y  ⇒  W
```

Line by line, mapped to the formula:
- `xt = transpose(x)` → Xᵀ.
- `xtx = matmul(xt, x)` → XᵀX.
- `reg = ...` builds **XᵀX + λI**. The trick `+ (lam if i == j else 0.0)` adds
  `lam` only where row index equals column index — i.e. on the **diagonal** —
  which is exactly "add λ times the identity matrix." That's the ridge penalty.
- `xty = matmul(xt, y)` → Xᵀ Y.
- Instead of literally computing the inverse `(...)⁻¹` (slow and unstable), we
  **solve** `reg · W = xty` for W — mathematically identical, numerically better.
- The returned `W` is the trained weight matrix. That's the entire "learning".

### Small helpers

```python
def clamp(value, lo, hi):  return max(lo, min(hi, value))
def sigmoid(x):            ...   # squashes any number into (0, 1)
```

- `clamp` — force a value to stay within [lo, hi]. `clamp(1.5, 0, 1)` → 1.0.
- `sigmoid` — the classic S-shaped squashing function `1/(1+e⁻ˣ)`. It's written
  in two branches to avoid `exp` overflow on large inputs (a numerical-safety
  detail). Not heavily used here but a standard tool worth recognising.

That's linalg.py. Everything is loops and lists — no magic. You now understand
what libraries like NumPy do internally.

---

## A.4 `sentinelx/config.py` — settings + a tiny YAML reader

### The default config

```python
DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {"name": ..., "dataset": "synthetic",
                   "model_type": "linear_transition", "seed": 1337},
    "data": {"num_windows": 40, "window_seconds": 60, "num_hosts": 18, ...},
    "graph": {...}, "model": {...}, "forecast": {"horizon": 3},
    "deviation": {"weights": {...}, "anomaly_threshold": 0.45, ...},
    "uncertainty": {...}, "novelty": {...}, "stability": {...},
    "api": {"host": "127.0.0.1", "port": 8787},
    "persistence": {"db_path": "sentinelx.db"},
}
```

- This nested dictionary is the **single source of truth** for every setting.
  A YAML file only *overrides* pieces of it.
- `dict[str, Any]` — "a dictionary with string keys and values of any type."
- Notice the comments recording *why* numbers are what they are (e.g. the
  deviation weights were calibrated to hit precision ~0.98). Good code explains
  its magic numbers.

### The `Config` class (typed accessor)

```python
@dataclass
class Config:
    data: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))

    def get(self, dotted: str, default=None):
        node = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node
```

- `@dataclass` — a **decorator** (a label starting with `@` that modifies the
  class below it). `@dataclass` auto-writes boilerplate like the constructor, so
  we don't hand-write `__init__`. Hugely used in this project.
- `field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))` — sets the
  default value of `data` to a *fresh deep copy* of the defaults. A **deep copy**
  duplicates nested structures so editing one Config can't accidentally mutate
  the shared defaults. A `lambda` is a tiny anonymous function; `default_factory`
  needs a function that *produces* the default (so each instance gets its own).
- `get("experiment.seed")` — a convenience: split the **dotted path** on `.` and
  walk into the nested dict step by step, returning `default` if any step is
  missing. So instead of `cfg.data["experiment"]["seed"]` (which crashes if a key
  is absent) you write `cfg.get("experiment.seed", 1337)` safely.

```python
    def section(self, name):  return self.data.get(name, {}) ...
    @property
    def seed(self):           return int(self.get("experiment.seed", 1337))
    def to_yaml(self):        return dump_yaml(self.data)
```

- `section("data")` returns a whole sub-dictionary.
- `@property` makes `seed` usable as `cfg.seed` (no parentheses) — it looks like
  an attribute but runs code. A clean way to expose a computed value.
- `to_yaml()` serialises the config back to YAML text (stored per experiment for
  reproducibility).

### Loading and merging configs

```python
def deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)   # recurse into sub-dicts
        else:
            out[key] = copy.deepcopy(value)
    return out
```

- **Merging** = start from defaults, then overlay the user's overrides. This is
  **recursive**: if both sides have a sub-dictionary for the same key, we merge
  *those* too (so overriding one threshold doesn't wipe the whole section).
- `isinstance(x, dict)` checks "is x a dictionary?". A function calling *itself*
  (`deep_merge` inside `deep_merge`) is **recursion** — perfect for nested data.

```python
def load_config(path=None, overrides=None):
    merged = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path) as fh:
            file_cfg = parse_yaml(fh.read())
        merged = deep_merge(merged, file_cfg or {})
    if overrides:
        merged = deep_merge(merged, overrides)
    return Config(merged)
```

The precedence is: **defaults → YAML file → explicit overrides** (each later one
wins). `with open(path) as fh:` is the standard safe way to read a file — the
`with` block automatically closes the file even if an error happens.

### The tiny YAML reader (why and how)

```python
def parse_yaml(text):
    try:
        from ruamel.yaml import YAML          # use a real library if present
        ...
    except Exception:
        pass
    try:
        import yaml as pyyaml                  # or PyYAML if present
        ...
    except Exception:
        pass
    return _parse_yaml_subset(text)           # else: our own parser
```

- `parse_yaml` first *tries* to use a real YAML library (`ruamel.yaml`, then
  `PyYAML`). `try/except` means "attempt this; if it fails, don't crash — do the
  next thing." If neither library is installed (our zero-dependency case), it
  falls back to our own `_parse_yaml_subset`.
- This is a nice pattern: **use the best tool available, degrade gracefully.**

`_parse_yaml_subset` reads the indentation-based subset of YAML we actually use
(nested `key: value` maps, `- item` lists, `#` comments) and coerces values to
the right types (`true`→bool, `42`→int, `0.3`→float, quoted→string). It's a small
state machine over the lines. You rarely need to touch it, but it proves a point:
even "parse a config format" is just careful string handling, not magic.

---

## Recap

- `__init__.py` makes a folder importable.
- `seeding.py` gives reproducible, independent random streams from one seed.
- `linalg.py` is NumPy-in-miniature: vectors, matrices, `matmul`, `solve`, and
  `ridge_fit` (the training maths), all as plain loops.
- `config.py` centralises every setting, merges defaults with overrides, and can
  read YAML with zero dependencies.

Next: [Code B — the data layer](02-data.md)
