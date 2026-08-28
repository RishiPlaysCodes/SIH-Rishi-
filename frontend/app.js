/* Sentinel-X dashboard controller.
 * Vanilla JS + D3 (force-directed graph + uncertainty chart). Talks to the
 * stdlib API on the same origin. Kept framework-free so it is served directly
 * by the Python HTTP server; the React/TS + Vite dashboard is the documented
 * swap-in and consumes the identical endpoint contract.
 */
"use strict";

const state = { window: null, numWindows: 1, view: "network", horizon: 1, forecast: null };
const COLORS = { normal: "#35e6ff", deviating: "#ffb347", anomalous: "#ff5470", server: "#9b6bff" };

const api = {
  async get(path) { const r = await fetch(path); if (!r.ok) throw new Error((await r.json()).error || r.statusText); return r.json(); },
  async post(path, body) { const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); if (!r.ok) throw new Error((await r.json()).error || r.statusText); return r.json(); },
};

function toast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 2600);
}

function nodeColor(n) { if (n.is_server) return COLORS.server; return COLORS[n.status] || COLORS.normal; }

/* ------------------------------------------------------------------ boot */
window.addEventListener("DOMContentLoaded", async () => {
  document.querySelectorAll(".tab").forEach((btn) =>
    btn.addEventListener("click", () => switchView(btn.dataset.view)));
  document.getElementById("window-slider").addEventListener("input", (e) => {
    state.window = +e.target.value; refresh();
  });
  document.getElementById("cf-run").addEventListener("click", runCounterfactual);

  try {
    const summary = await api.get("/summary");
    renderSummary(summary);
    state.numWindows = summary.num_windows || 40;
  } catch (e) { toast("API not reachable: " + e.message); return; }

  const slider = document.getElementById("window-slider");
  slider.max = state.numWindows - 1;
  slider.value = state.numWindows - 1;
  state.window = state.numWindows - 1;
  await refresh();
});

function renderSummary(s) {
  document.getElementById("stat-model").textContent = s.model_type || "—";
  const d = s.detection || {};
  document.getElementById("stat-precision").textContent = d.precision != null ? d.precision.toFixed(2) : "—";
  document.getElementById("stat-recall").textContent = d.recall != null ? d.recall.toFixed(2) : "—";
  document.getElementById("stat-windows").textContent = s.num_windows || "—";
}

function switchView(view) {
  state.view = view;
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.dataset.view === view));
  refresh();
}

async function refresh() {
  document.getElementById("window-label").textContent =
    `${state.window} / ${state.numWindows - 1}`;
  try {
    if (state.view === "network") await renderNetwork();
    else if (state.view === "forecast") await renderForecast();
    else if (state.view === "uncertainty") await renderUncertainty();
    else if (state.view === "propagation") await renderPropagation();
    else if (state.view === "counterfactual") await renderCounterfactualControls();
    else if (state.view === "chronicle") await renderChronicle();
  } catch (e) { toast(e.message); }
}

/* ------------------------------------------------------------ NETWORK */
async function renderNetwork() {
  const data = await api.get(`/network/state?window=${state.window}`);
  drawGraph("#network-graph", data.nodes, data.edges, onNodeClick);
  const list = document.getElementById("anomaly-list");
  list.innerHTML = "";
  (data.anomalies || []).slice(0, 8).forEach((a) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="k">${a.node}</span>
      <span><span class="badge ${a.status}">${a.status}</span>
      <span class="v">${(a.deviation_score * 100).toFixed(0)}%</span></span>`;
    list.appendChild(li);
  });
  // populate counterfactual targets from current nodes
  const sel = document.getElementById("cf-target");
  if (sel && sel.dataset.win !== String(state.window)) {
    sel.innerHTML = "";
    data.nodes.forEach((n) => { const o = document.createElement("option"); o.value = n.key; o.textContent = n.label; sel.appendChild(o); });
    sel.dataset.win = String(state.window);
    const anom = (data.anomalies || []).find((a) => a.status === "anomalous");
    if (anom) sel.value = anom.node;
  }
  window._netFeatureNames = null;
  window._netNodes = data.nodes;
}

function onNodeClick(n) {
  document.getElementById("node-hint").style.display = "none";
  const featNames = ["conn_freq","uniq_dst","uniq_ports","failed_conn","out_ratio","pkt_rate","byte_rate","mean_iat"];
  const detail = document.getElementById("node-detail");
  const max = Math.max(1, ...n.features.map((f) => Math.abs(f)));
  detail.innerHTML = `<div class="feat"><b>${n.label}</b><span class="badge ${n.status}">${n.status}</span></div>` +
    n.features.map((f, i) => {
      const w = Math.min(100, (Math.abs(f) / max) * 100);
      return `<div class="feat"><span>${featNames[i] || "f" + i}</span>
        <span style="display:flex;align-items:center;gap:8px">
        <span class="bar" style="width:${w}px;background:${f >= 0 ? "#35e6ff" : "#ff5470"}"></span>
        ${f.toFixed(2)}</span></div>`;
    }).join("");
}

/* ------------------------------------------------------------ FORECAST */
async function renderForecast() {
  const fc = await api.get(`/forecast?window=${state.window}`);
  state.forecast = fc;
  const tabs = document.getElementById("horizon-tabs");
  tabs.innerHTML = "";
  fc.steps.forEach((s) => {
    const b = document.createElement("button");
    b.textContent = "T+" + s.horizon;
    b.classList.toggle("active", s.horizon === state.horizon);
    b.addEventListener("click", () => { state.horizon = s.horizon; drawForecastStep(); });
    tabs.appendChild(b);
  });
  if (!fc.steps.find((s) => s.horizon === state.horizon)) state.horizon = fc.steps[0].horizon;
  drawForecastStep();
}

function drawForecastStep() {
  const step = state.forecast.steps.find((s) => s.horizon === state.horizon) || state.forecast.steps[0];
  document.querySelectorAll("#horizon-tabs button").forEach((b) =>
    b.classList.toggle("active", b.textContent === "T+" + step.horizon));
  drawGraph("#forecast-graph", step.graph.nodes, step.graph.edges, onNodeClick, true);
  document.getElementById("forecast-detail").innerHTML =
    `<ul class="metric-list">
      <li><span class="k">Horizon</span><span class="v">T+${step.horizon}</span></li>
      <li><span class="k">Uncertainty σ</span><span class="v">${step.uncertainty_sigma.toFixed(3)}</span></li>
      <li><span class="k">Confidence</span><span class="badge ${step.uncertainty_label === 'HIGH' ? 'anomalous' : step.uncertainty_label === 'MEDIUM' ? 'deviating' : 'normal'}">${step.uncertainty_label}</span></li>
      <li><span class="k">Predicted nodes</span><span class="v">${step.graph.nodes.length}</span></li>
    </ul>`;
}

/* --------------------------------------------------------- UNCERTAINTY */
async function renderUncertainty() {
  const fc = await api.get(`/forecast?window=${state.window}`);
  const svg = d3.select("#uncertainty-chart");
  svg.selectAll("*").remove();
  const W = svg.node().clientWidth || 900, H = 380, m = { t: 20, r: 20, b: 40, l: 50 };
  const pts = fc.steps.map((s) => ({ x: s.horizon, sigma: s.uncertainty_sigma, mean: 0.5 }));
  const x = d3.scaleLinear().domain([1, d3.max(pts, (d) => d.x)]).range([m.l, W - m.r]);
  const maxS = d3.max(pts, (d) => d.mean + d.sigma) * 1.2;
  const y = d3.scaleLinear().domain([0, maxS]).range([H - m.b, m.t]);

  svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(pts.length).tickFormat((d) => "T+" + d)).attr("color", "#8797c4");
  svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y)).attr("color", "#8797c4");

  const area = d3.area().x((d) => x(d.x)).y0((d) => y(Math.max(0, d.mean - d.sigma))).y1((d) => y(d.mean + d.sigma)).curve(d3.curveMonotoneX);
  svg.append("path").datum(pts).attr("fill", "rgba(155,107,255,.25)").attr("d", area);
  const line = d3.line().x((d) => x(d.x)).y((d) => y(d.mean)).curve(d3.curveMonotoneX);
  svg.append("path").datum(pts).attr("fill", "none").attr("stroke", "#35e6ff").attr("stroke-width", 2).attr("d", line);
  svg.selectAll("circle.pt").data(pts).enter().append("circle").attr("class", "pt")
    .attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.mean)).attr("r", 4).attr("fill", "#35e6ff");
  pts.forEach((p) => {
    svg.append("text").attr("x", x(p.x)).attr("y", y(p.mean + p.sigma) - 8)
      .attr("fill", "#8797c4").attr("text-anchor", "middle").attr("font-size", 11)
      .text("σ=" + p.sigma.toFixed(2));
  });
}

/* --------------------------------------------------------- PROPAGATION */
async function renderPropagation() {
  const [netw, prop] = await Promise.all([
    api.get(`/network/state?window=${state.window}`),
    api.get(`/propagation`),
  ]);
  const propEdges = prop.events.map((e) => ({ src: e.source, dst: e.target, anomaly: true }));
  const merged = netw.edges.concat(propEdges);
  drawGraph("#propagation-graph", netw.nodes, merged, onNodeClick);
  const list = document.getElementById("propagation-metrics");
  list.innerHTML = "";
  if (!prop.events.length) { list.innerHTML = `<li class="hint">No propagation detected yet.</li>`; return; }
  const last = prop.events[prop.events.length - 1];
  const rows = [
    ["Events", prop.events.length],
    ["Latest path", `${last.source} → ${last.target}`],
    ["Velocity", last.propagation_velocity.toFixed(3) + " /s"],
    ["Intensity", last.propagation_intensity.toFixed(3)],
    ["Effective R", last.effective_reproduction_number.toFixed(2)],
  ];
  rows.forEach(([k, v]) => { const li = document.createElement("li"); li.innerHTML = `<span class="k">${k}</span><span class="v">${v}</span>`; list.appendChild(li); });
}

/* ------------------------------------------------------ COUNTERFACTUAL */
async function renderCounterfactualControls() {
  const data = await api.get(`/network/state?window=${state.window}`);
  const sel = document.getElementById("cf-target");
  sel.innerHTML = "";
  data.nodes.forEach((n) => { const o = document.createElement("option"); o.value = n.key; o.textContent = n.label; sel.appendChild(o); });
  const anom = (data.anomalies || []).find((a) => a.status === "anomalous");
  if (anom) sel.value = anom.node;
}

async function runCounterfactual() {
  const body = {
    action_type: document.getElementById("cf-action").value,
    window: state.window,
    target_node: document.getElementById("cf-target").value,
    port: +document.getElementById("cf-port").value,
  };
  let res;
  try { res = await api.post("/counterfactual", body); } catch (e) { toast(e.message); return; }
  const cb = res.components_before, ca = res.components_after;
  const rows = [
    ["DB / server risk", cb.db_risk, ca.db_risk],
    ["Lateral movement risk", cb.lateral_movement_risk, ca.lateral_movement_risk],
    ["Alternate path risk", cb.alternate_path_risk, ca.alternate_path_risk],
    ["Overall risk", cb.overall, ca.overall],
  ];
  document.getElementById("cf-body").innerHTML = rows.map(([k, b, a]) => {
    const d = b - a; const cls = d > 0.001 ? "down" : d < -0.001 ? "up" : "";
    return `<tr><td>${k}</td><td>${(b * 100).toFixed(0)}%</td><td>${(a * 100).toFixed(0)}%</td>
      <td class="${cls}">${d >= 0 ? "−" : "+"}${Math.abs(d * 100).toFixed(0)}%</td></tr>`;
  }).join("");
  toast(`ΔRisk ${(res.delta_risk * 100).toFixed(0)}% via ${res.action_type}`);
}

/* --------------------------------------------------------- CHRONICLE */
async function renderChronicle() {
  const data = await api.get("/incident");
  const feed = document.getElementById("chronicle-feed");
  feed.innerHTML = "";
  if (!data.incidents.length) { feed.innerHTML = `<li class="hint">No incidents recorded.</li>`; return; }
  data.incidents.forEach((i) => {
    const li = document.createElement("li");
    li.className = i.event_type;
    li.innerHTML = `<div class="evt">${i.event_type.replace(/_/g, " ")}</div>
      <div class="txt">${i.narrative_text}</div>
      ${i.mitre_stage ? `<span class="stage">ATT&CK · ${i.mitre_stage}</span>` : ""}`;
    feed.appendChild(li);
  });
}

/* --------------------------------------------------------- D3 GRAPH */
function drawGraph(selector, nodes, edges, onClick, forecast) {
  const svg = d3.select(selector);
  svg.selectAll("*").remove();
  const W = svg.node().clientWidth || 800, H = svg.node().clientHeight || 560;
  const nodeById = new Map(nodes.map((n) => [n.key, n]));
  const links = edges
    .filter((e) => nodeById.has(e.src) && nodeById.has(e.dst))
    .map((e) => ({ source: e.src, target: e.dst, anomaly: !!e.anomaly, weight: e.weight || 1 }));

  const sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((d) => d.key).distance(70).strength(.4))
    .force("charge", d3.forceManyBody().strength(-220))
    .force("center", d3.forceCenter(W / 2, H / 2))
    .force("collide", d3.forceCollide(22));

  const defs = svg.append("defs");
  defs.append("marker").attr("id", "arrow").attr("viewBox", "0 -5 10 10").attr("refX", 20)
    .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
    .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#2a3a6e");

  const link = svg.append("g").selectAll("line").data(links).enter().append("line")
    .attr("class", (d) => "link" + (d.anomaly ? " anomaly" : ""))
    .attr("stroke-width", (d) => Math.min(4, 1 + Math.log10(d.weight + 1)))
    .attr("marker-end", "url(#arrow)");

  const g = svg.append("g").selectAll("g").data(nodes).enter().append("g").attr("class", (d) => "node" + (d.status === "anomalous" ? " pulse" : ""))
    .call(d3.drag()
      .on("start", (e, d) => { if (!e.active) sim.alphaTarget(.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));

  g.each(function (d) {
    const sel = d3.select(this);
    if (d.is_server) sel.append("rect").attr("width", 20).attr("height", 20).attr("x", -10).attr("y", -10).attr("rx", 3).attr("fill", nodeColor(d));
    else sel.append("circle").attr("r", (d.status === "anomalous" ? 11 : 8)).attr("fill", nodeColor(d));
  });
  g.append("text").attr("x", 14).attr("y", 4).text((d) => d.label);
  g.style("opacity", forecast ? 0.92 : 1);
  g.on("click", (e, d) => onClick && onClick(d));
  g.append("title").text((d) => `${d.label} (${d.status})`);

  sim.on("tick", () => {
    link.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
    g.attr("transform", (d) => `translate(${d.x},${d.y})`);
  });
}
