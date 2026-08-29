# Code J · The Dashboard (`frontend/`)

Pipeline **step 11**: the browser UI. Three files, and **zero build step** — it's
plain HTML, CSS, and JavaScript, plus **D3.js** loaded from a CDN. The stdlib
server (Part I) serves these files. (The industrial swap-in is React + TypeScript
+ Vite + Tailwind.)

> **Web crash course:** a web page is three languages working together.
> **HTML** = the *structure* (what elements exist). **CSS** = the *style* (colours,
> layout). **JavaScript (JS)** = the *behaviour* (fetch data, react to clicks,
> draw things). The browser runs all three.

---

## J.1 `index.html` — the structure

```html
<head>
  <link rel="stylesheet" href="/styles.css" />
  <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js" defer></script>
  <script src="/app.js" defer></script>
</head>
```

- `<link rel="stylesheet" ...>` pulls in our CSS.
- The first `<script>` loads **D3.js** from a **CDN** (Content Delivery Network —
  a public host for popular libraries). This is the *only* thing fetched from the
  internet, and it happens in *your* browser. D3 is the library we use to draw the
  interactive network graph.
- `defer` means "download now, run after the page's HTML is parsed" — so the
  script doesn't run before the elements it needs exist.
- The second `<script>` is our own `app.js`.

```html
<nav class="tabs" id="tabs">
  <button class="tab active" data-view="network">Network</button>
  <button class="tab" data-view="forecast">Forecast</button>
  ... Uncertainty / Propagation / Counterfactual / CyberChronicle ...
</nav>

<div class="scrubber">
  <input type="range" id="window-slider" min="0" max="0" value="0" />
</div>

<main id="views">
  <section class="view active" data-view="network">
    <svg id="network-graph"></svg>
    <aside class="panel"> ... node inspector ... </aside>
  </section>
  ... one <section> per view ...
</main>
```

- `<button>`s are the six tabs. `data-view="network"` is a **data attribute** — a
  custom label JS reads to know which view a button switches to.
- `<input type="range">` is the **time-window scrubber** (a slider).
- Each `<section class="view">` is one screen. `<svg>` is a **Scalable Vector
  Graphics** canvas — the blank surface D3 draws the graph onto.
- `id="..."` gives elements names so JS/CSS can find them.

---

## J.2 `styles.css` — the "command-center" look

CSS is a list of rules: *"for elements matching this selector, set these
properties."*

```css
:root {
  --bg: #070b18;
  --cyan: #35e6ff;
  --amber: #ffb347;
  --red: #ff5470;
  ...
}
body { background: ...; color: var(--text); font-family: var(--sans); }
.tab.active { color: var(--cyan); border-bottom-color: var(--cyan); }
.dot.anomalous { background: var(--red); }
```

- `:root { --bg: ...; }` defines **CSS variables** (reusable values). `var(--cyan)`
  uses one. Defining the palette once keeps the whole UI consistent — the dark
  navy background, glowing cyan for normal, amber for "deviating", red for
  "anomalous", violet for servers.
- `.tab.active` styles a tab that has *both* classes `tab` and `active` (the
  selected one). `.dot.anomalous` colours the anomalous-status legend dot red.
- There are also `@keyframes` animations (a subtle pulse on active/anomalous
  nodes) — motion that communicates *state*, not decoration, matching the design
  spec.

You don't need to memorise CSS; just know it maps selectors → visual properties,
and our variables encode the status colour language (normal/deviating/anomalous).

---

## J.3 `app.js` — the behaviour

### Talking to the API

```javascript
const api = {
  async get(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error((await r.json()).error || r.statusText);
    return r.json();
  },
  async post(path, body) {
    const r = await fetch(path, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error((await r.json()).error || r.statusText);
    return r.json();
  },
};
```

- `fetch(path)` makes an HTTP request from the browser to our server — the same
  URLs from Part I. Because the dashboard is served *by* that server, `fetch("/summary")`
  hits the right place automatically (same origin).
- `async`/`await` handle the fact that network calls take time: `await` pauses
  until the response arrives without freezing the page. `r.json()` parses the JSON
  body into a JS object. If the status isn't OK, we throw an error (shown as a
  toast). This tiny object is our whole "API client."

### The main loop

```javascript
async function refresh() {
  document.getElementById("window-label").textContent = `${state.window} / ${state.numWindows-1}`;
  if (state.view === "network")       await renderNetwork();
  else if (state.view === "forecast") await renderForecast();
  ...
}
```

- A single `state` object tracks the current tab and window. `refresh()` looks at
  `state.view` and calls the matching render function. Clicking a tab or moving
  the slider updates `state` and calls `refresh()`. Simple, framework-free state
  management.

```javascript
async function renderNetwork() {
  const data = await api.get(`/network/state?window=${state.window}`);
  drawGraph("#network-graph", data.nodes, data.edges, onNodeClick);
  // fill the "top deviations" side panel from data.anomalies ...
}
```

- Fetch the window's graph from the API, draw it, and populate the side panel.
  Every view follows this shape: **fetch JSON → render**.

### Drawing the graph with D3 (a force simulation)

```javascript
function drawGraph(selector, nodes, edges, onClick) {
  const svg = d3.select(selector);
  svg.selectAll("*").remove();                       // clear previous drawing
  const links = edges.filter(...).map(e => ({ source: e.src, target: e.dst, ... }));

  const sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.key).distance(70))
    .force("charge", d3.forceManyBody().strength(-220))
    .force("center", d3.forceCenter(W/2, H/2))
    .force("collide", d3.forceCollide(22));
```

- `d3.select` grabs the `<svg>`; `selectAll("*").remove()` clears any old drawing.
- **Force simulation** is D3's physics engine for graphs. It treats nodes like
  charged particles and edges like springs, then lets them settle into a
  readable layout:
  - `forceLink` — edges pull connected nodes together (spring).
  - `forceManyBody` with negative strength — nodes **repel** each other so they
    spread out.
  - `forceCenter` — gently pull everything toward the middle.
  - `forceCollide` — stop nodes from overlapping.
- The result is that classic "web of dots that arranges itself" you see on screen.

```javascript
  const g = svg.append("g").selectAll("g").data(nodes).enter().append("g")
    .attr("class", d => "node" + (d.status === "anomalous" ? " pulse" : ""))
    .call(d3.drag()...);                              // make nodes draggable
  g.each(function (d) {
    if (d.is_server) d3.select(this).append("rect")...fill(nodeColor(d));
    else             d3.select(this).append("circle").attr("r", ...).attr("fill", nodeColor(d));
  });
  sim.on("tick", () => {                              // every physics frame:
    link.attr("x1", d => d.source.x)...;             // move the lines
    g.attr("transform", d => `translate(${d.x},${d.y})`);  // move the nodes
  });
}
```

- **The `.data(nodes).enter().append(...)` pattern is the heart of D3:** it binds
  an array of data to SVG elements, creating one element per data item. This is
  the "data join" — D3's core idea (React later borrowed the same spirit).
- Servers are drawn as squares (`rect`), hosts as circles. `nodeColor(d)` returns
  the status colour (cyan/amber/red/violet), so the graph *shows* the detection at
  a glance.
- Anomalous nodes get the `pulse` class → the CSS animation makes them throb.
- `sim.on("tick", ...)` runs every animation frame: the physics engine updates
  each node's `x,y`, and we move the SVG shapes to match. That's what makes the
  layout *animate* into place and respond to dragging.

### The other views (same pattern)

- **Forecast** — fetch `/forecast`, draw the predicted graph for the chosen
  horizon step, show its uncertainty label.
- **Uncertainty** — draw a line chart with a shaded band that *widens* with the
  horizon (using `d3.area` for the band and `d3.line` for the mean).
- **Propagation** — draw the graph plus red animated edges for the infection path,
  and list velocity / intensity / Rₑ.
- **Counterfactual** — a form (pick action + target) that POSTs to
  `/counterfactual` and fills a before/after risk table, colouring drops green and
  rises red.
- **CyberChronicle** — fetch `/incident` and render the templated sentences as a
  feed, tagged with the MITRE stage.

Every one is: *fetch JSON → render with plain DOM or D3.* No framework needed to
understand it.

---

## Recap

Step 11 done. The dashboard is HTML (structure) + CSS (the dark status-coloured
theme) + vanilla JS (behaviour), with D3 doing the force-directed graph and the
uncertainty chart. It talks to the API with `fetch`, and every screen follows the
same "fetch JSON → render" recipe. You now understand the *whole* stack, browser
to database.

Next: [Code K — the tests](11-tests.md)
