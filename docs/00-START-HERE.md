# 0 · Start Here

## Who this book is for

- A student who has written a little Python (loops, functions, maybe a class)
  and wants to understand a *real* project end to end.
- Someone who has heard words like "machine learning", "graph", "API",
  "anomaly detection" but never built something that uses all of them together.

You do **not** need to know: machine learning, linear algebra, networking,
web development, or databases. We teach each as it comes up.

## How to get the most out of it

1. **Read in order.** Later parts lean on earlier ones.
2. **Type the code yourself.** Do not copy-paste. Typing forces your brain to
   read every character. This is how you actually learn.
3. **Run things constantly.** After each code walkthrough, run the file or the
   tests. Seeing output cements understanding.
4. **When you hit a new word,** it will be in **bold** the first time — that is
   your signal that a definition follows right there. If you forget one later,
   the [glossary](07-glossary.md) has them all.

## Set up your machine (one time)

You need **Python 3.10 or newer**. Check what you have:

```bash
python --version
# or on some systems:
python3 --version
```

If it prints `Python 3.10.x` or higher, you are ready. If not, install Python
from [python.org](https://www.python.org/downloads/).

That's it. **There is nothing else to install** — this project uses only Python's
built-in "standard library". (We explain what that means in
[Part 3](03-tech-stack.md).)

### Get the code

```bash
git clone https://github.com/RishiPlaysCodes/SIH-Rishi-.git
cd SIH-Rishi-
```

### Prove it runs (your first win)

```bash
python -m sentinelx.cli run
```

You should see a block of JSON describing a run — number of time windows, a
detection score, etc. If you see that, everything works and you're ready to
learn how it works.

```bash
# See the live dashboard in your browser:
python -m sentinelx.cli serve
# then open http://127.0.0.1:8787
```

## The mental model to hold onto

Everything in this project is one pipeline (a "pipe" that data flows through):

```
raw network traffic
      │
      ▼   turn it into numbers (features)
   a graph for each slice of time
      │
      ▼   learn "what normal looks like"
   a model that predicts the next moment
      │
      ▼   compare prediction vs reality
   an "how weird is this?" score
      │
      ▼   dress it up for a human
   dashboard + plain-English incident log
```

Keep this picture in your head. Every file we read fits somewhere on this pipe.

Next: [Part 1 — What is Sentinel-X](01-what-is-sentinelx.md)
