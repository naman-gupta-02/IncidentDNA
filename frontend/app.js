/* IncidentDNA dashboard.
 *
 * Dependency-free on purpose: no CDN, no build step, so `make demo` works
 * offline and the repo has nothing to install for the UI. Charts and the
 * service graph are hand-drawn SVG.
 */
"use strict";

const API = "";
const POLL_STATE_MS = 1500;
const POLL_DETAIL_MS = 3000;

const state = {
  topology: null,
  faults: [],
  snapshot: null,
  incidents: [],
  selectedIncidentId: null,
  followLive: true,
  incident: null,
  metrics: null,
  activeTab: "timeline",
  evaluation: undefined,
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return node;
};

async function api(path, options) {
  const res = await fetch(API + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.headers.get("content-type")?.includes("json") ? res.json() : res.text();
}

const fmtMs = (v) => (v >= 1000 ? (v / 1000).toFixed(2) + " s" : Math.round(v) + " ms");
const fmtPct = (v) => (v * 100).toFixed(1) + "%";
const fmtNum = (v) => (Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : v.toFixed(2));
const clockOf = (seconds) => {
  const d = new Date(seconds * 1000);
  return d.toISOString().slice(11, 19);
};

/* ---------------------------------------------------------------- controls */
async function initControls() {
  state.topology = await api("/api/topology");
  state.faults = await api("/api/faults");

  const faultSel = $("#fault-select");
  faultSel.replaceChildren(
    ...state.faults.map((f) => el("option", { value: f.name, title: f.description }, f.description))
  );
  const syncOrigins = () => {
    const f = state.faults.find((x) => x.name === faultSel.value) || state.faults[0];
    $("#origin-select").replaceChildren(
      ...f.origins.map((o) =>
        el("option", { value: o }, state.topology.services.find((s) => s.name === o)?.display_name || o)
      )
    );
  };
  faultSel.addEventListener("change", syncOrigins);
  syncOrigins();

  const sev = $("#severity");
  sev.addEventListener("input", () => ($("#severity-value").textContent = sev.value));

  $("#inject").addEventListener("click", async (e) => {
    e.target.disabled = true;
    try {
      await api("/api/faults/inject", {
        method: "POST",
        body: JSON.stringify({
          fault_type: faultSel.value,
          origin: $("#origin-select").value,
          severity: parseFloat(sev.value),
          duration_seconds: 300,
        }),
      });
      state.followLive = true;
      await refreshState();
    } catch (err) {
      alert("Injection failed: " + err.message);
    } finally {
      e.target.disabled = false;
    }
  });

  $("#clear").addEventListener("click", async () => {
    await api("/api/faults/clear", { method: "POST" });
    await refreshState();
  });

  $("#ranker-select").addEventListener("change", async (e) => {
    await api("/api/ranker", { method: "POST", body: JSON.stringify({ ranker: e.target.value }) });
    await refreshState();
    await refreshDetail();
  });

  $("#tabs").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-tab]");
    if (!btn) return;
    state.activeTab = btn.dataset.tab;
    document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b === btn));
    document
      .querySelectorAll(".tabpane")
      .forEach((p) => p.classList.toggle("active", p.id === "tab-" + state.activeTab));
    if (state.activeTab === "evaluation" && state.evaluation === undefined) loadEvaluation();
  });
}

/* ------------------------------------------------------------------ polling */
async function refreshState() {
  state.snapshot = await api("/api/state");
  state.incidents = await api("/api/incidents");
  if (state.followLive && state.snapshot.open_incident_id) {
    state.selectedIncidentId = state.snapshot.open_incident_id;
  } else if (!state.selectedIncidentId && state.incidents.length) {
    state.selectedIncidentId = state.incidents[0].incident_id;
  }
  renderStatus();
  renderGraph();
  renderIncidentList();
}

async function refreshDetail() {
  if (state.selectedIncidentId) {
    try {
      state.incident = await api("/api/incidents/" + state.selectedIncidentId);
    } catch {
      state.incident = null;
    }
  } else {
    state.incident = null;
  }
  const services = (state.topology?.services || []).map((s) => s.name).join(",");
  state.metrics = await api(`/api/metrics?services=${services}&limit=160`);
  renderDiagnosis();
  renderTimeline();
  renderMetrics();
  renderTraces();
  renderCandidates();
}

/* ------------------------------------------------------------------- header */
function renderStatus() {
  const s = state.snapshot;
  if (!s) return;
  $("#stat-source").textContent = s.source === "kafka" ? "Kafka (live services)" : `simulated ${s.speed}x`;
  $("#stat-clock").textContent = clockOf(s.simulated_clock);
  $("#stat-windows").textContent = s.windows_processed.toLocaleString();
  $("#stat-records").textContent = s.telemetry_records.toLocaleString();
  $("#stat-incidents").textContent = s.incident_count;

  const sel = $("#ranker-select");
  if (sel.options.length !== s.available_rankers.length) {
    sel.replaceChildren(...s.available_rankers.map((r) => el("option", { value: r }, r)));
  }
  sel.value = s.ranker;

  const af = $("#active-faults");
  if (s.active_faults.length) {
    const f = s.active_faults[0];
    af.className = "pill live";
    af.textContent = `${f.fault_type} @ ${f.origin_display} · ${f.remaining_seconds.toFixed(0)}s left`;
  } else {
    af.className = "pill muted";
    af.textContent = "no active fault";
  }
}

/* -------------------------------------------------------------------- graph */
const GRAPH_LAYOUT = {
  "api-gateway": [0.02, 0.50],
  "order-service": [0.42, 0.50],
  "payment-service": [0.90, 0.10],
  "inventory-service": [0.90, 0.50],
  postgres: [0.90, 0.90],
};

function renderGraph() {
  const s = state.snapshot;
  if (!s || !state.topology) return;
  const W = 820;
  const H = 350;
  // Wide enough for the longest service name ("Inventory Service") and the
  // score to share the title row with clearance to spare.
  const NW = 204;
  const NH = 60;
  const byName = Object.fromEntries(s.services.map((x) => [x.service, x]));
  const pos = (name) => {
    const [fx, fy] = GRAPH_LAYOUT[name] || [0.5, 0.5];
    return { x: fx * (W - NW) + NW / 2, y: fy * (H - NH) + NH / 2 };
  };
  const colorOf = (svc) =>
    svc.is_root_cause
      ? "var(--cause)"
      : svc.health === "critical"
      ? "var(--critical)"
      : svc.health === "degraded"
      ? "var(--degraded)"
      : "var(--healthy)";

  const parts = [];
  parts.push(`<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
      markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#31507f"/></marker></defs>`);

  // Where a ray from a node's centre leaves its box. Scaling the direction
  // vector by a fixed radius (the old approach) detaches diagonal edges from
  // their boxes as soon as the nodes are wider than they are tall.
  const edgeOf = (centre, ux, uy, pad) => {
    const hw = NW / 2 + pad;
    const hh = NH / 2 + pad;
    const t = Math.min(
      Math.abs(ux) < 1e-6 ? Infinity : hw / Math.abs(ux),
      Math.abs(uy) < 1e-6 ? Infinity : hh / Math.abs(uy)
    );
    return { x: centre.x + ux * t, y: centre.y + uy * t };
  };

  for (const e of state.topology.edges) {
    const a = pos(e.source);
    const b = pos(e.target);
    const len = Math.hypot(b.x - a.x, b.y - a.y) || 1;
    const ux = (b.x - a.x) / len;
    const uy = (b.y - a.y) / len;
    const from = edgeOf(a, ux, uy, 4);
    const to = edgeOf(b, -ux, -uy, 10);
    const sx = from.x, sy = from.y, tx = to.x, ty = to.y;
    const hot = byName[e.source]?.health !== "healthy" && byName[e.target]?.health !== "healthy";
    parts.push(
      `<line x1="${sx}" y1="${sy}" x2="${tx}" y2="${ty}" stroke="${hot ? "#f87171" : "#31507f"}"
        stroke-width="${hot ? 2.4 : 1.6}" marker-end="url(#arrow)" opacity="${hot ? 0.9 : 0.65}"/>`
    );
  }

  for (const node of state.topology.services) {
    const svc = byName[node.name] || { health: "healthy", score: 0, metrics: {} };
    const p = pos(node.name);
    const stroke = colorOf(svc);
    const x = p.x - NW / 2;
    const y = p.y - NH / 2;
    const p95 = svc.metrics?.latency_p95;
    const err = svc.metrics?.error_rate;
    // The score sits on the title row, right-aligned, and the metrics get the
    // whole of the second row. Sharing one baseline made the two collide as
    // soon as a service had both a four-digit p95 and a visible error rate.
    const metrics = [
      p95 != null ? `p95 ${fmtMs(p95)}` : null,
      err >= 0.005 ? `err ${(err * 100).toFixed(1)}%` : null,
    ]
      .filter(Boolean)
      .join("  ·  ");
    // The title is clipped short of the score rather than trusted to fit:
    // glyph advances differ across the system font stack, so a layout that is
    // merely "wide enough" here can still collide on someone else's machine.
    const clipId = `clip-${node.name}`;
    const titleRight = NW - 46;
    parts.push(`
      <g>
        <clipPath id="${clipId}">
          <rect x="${x}" y="${y}" width="${titleRight}" height="${NH}"/>
        </clipPath>
        <rect x="${x}" y="${y}" width="${NW}" height="${NH}" rx="9"
              fill="#16243d" stroke="${stroke}" stroke-width="${svc.is_root_cause ? 3 : 1.6}"/>
        ${svc.is_root_cause ? `<rect x="${x - 5}" y="${y - 5}" width="${NW + 10}" height="${NH + 10}" rx="12"
              fill="none" stroke="var(--cause)" stroke-width="1" opacity=".45"/>` : ""}
        <circle cx="${x + 14}" cy="${y + 20}" r="4.5" fill="${stroke}"/>
        <text x="${x + 26}" y="${y + 24}" fill="#e6edf7" font-size="12.5" font-weight="620"
              clip-path="url(#${clipId})">${node.display_name}</text>
        <text x="${x + NW - 13}" y="${y + 24}" fill="${stroke}" font-size="11.5" text-anchor="end"
              font-weight="700">${svc.score.toFixed(2)}</text>
        <text x="${x + 14}" y="${y + 43}" fill="#93a4c0" font-size="10.5">${metrics}</text>
        ${svc.is_root_cause ? `<text x="${x + NW / 2}" y="${y - 13}" fill="var(--cause)" font-size="10"
              text-anchor="middle" font-weight="700" letter-spacing=".08em">ROOT CAUSE</text>` : ""}
      </g>`);
  }
  $("#graph").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="service dependency graph">${parts.join("")}</svg>`;
}

/* ---------------------------------------------------------------- diagnosis */
function renderDiagnosis() {
  const box = $("#diagnosis");
  const inc = state.incident;
  $("#diagnosis-ranker").textContent = state.snapshot ? `ranker: ${state.snapshot.ranker}` : "";
  if (!inc) {
    box.replaceChildren(el("p", { class: "empty" }, "No incident selected. Inject a fault to start an investigation."));
    return;
  }
  const d = inc.diagnosis;
  const top = d?.candidates?.[0];
  const kids = [
    el(
      "div",
      { class: "diag-head" },
      el("span", { class: "diag-cause" }, top ? top.display_name : "Diagnosing…"),
      el("span", { class: "sev " + inc.severity_label }, inc.severity_label),
      el("span", { class: "pill muted" }, inc.incident_id),
      el("span", { class: "pill muted" }, inc.status)
    ),
    el("p", { class: "symptom" }, inc.symptom ? inc.symptom.text : "—"),
  ];
  if (top) {
    kids.push(
      el("div", { class: "hint" }, `Confidence ${(top.confidence * 100).toFixed(0)}% · detected ${inc.detection_delay_seconds}s after onset · started ${inc.started_clock}`),
      el("div", { class: "confidence-bar" }, el("i", { style: `width:${(top.confidence * 100).toFixed(1)}%` })),
      el("div", { class: "subhead" }, "Evidence"),
      el("ol", { class: "evidence" }, ...top.evidence.map((t) => el("li", {}, t)))
    );
    if (top.contradicting.length) {
      kids.push(
        el("div", { class: "subhead" }, "Contradicting evidence"),
        el("ol", { class: "evidence against" }, ...top.contradicting.map((t) => el("li", {}, t)))
      );
    }
  }
  box.replaceChildren(...kids);
}

/* ----------------------------------------------------------------- timeline */
function renderTimeline() {
  const pane = $("#tab-timeline");
  const inc = state.incident;
  if (!inc || !inc.timeline.length) {
    pane.replaceChildren(el("p", { class: "empty" }, "No incident selected."));
    return;
  }
  pane.replaceChildren(
    el(
      "ul",
      { class: "timeline" },
      ...inc.timeline.map((t) =>
        el("li", { class: t.kind }, el("span", { class: "ts" }, t.clock), t.text)
      )
    )
  );
}

/* ------------------------------------------------------------------- charts */
const SERIES_COLORS = ["#2dd4bf", "#a78bfa", "#fbbf24", "#f87171", "#60a5fa"];

function lineChart(title, windows, series, formatter) {
  const W = 420;
  const H = 150;
  const pad = { l: 46, r: 8, t: 8, b: 18 };
  const names = Object.keys(series);
  const all = names.flatMap((n) => series[n]).filter((v) => Number.isFinite(v));
  const max = Math.max(1e-9, ...all);
  const min = Math.min(0, ...all);
  const n = windows.length || 1;
  const xOf = (i) => pad.l + (i / Math.max(1, n - 1)) * (W - pad.l - pad.r);
  const yOf = (v) => H - pad.b - ((v - min) / (max - min || 1)) * (H - pad.t - pad.b);

  const grid = [0, 0.5, 1]
    .map((f) => {
      const v = min + f * (max - min);
      return `<line x1="${pad.l}" y1="${yOf(v)}" x2="${W - pad.r}" y2="${yOf(v)}" stroke="#1f3355" stroke-width="1"/>
              <text x="${pad.l - 6}" y="${yOf(v) + 3.5}" fill="#64748b" font-size="9" text-anchor="end">${formatter(v)}</text>`;
    })
    .join("");

  const paths = names
    .map((name, i) => {
      const pts = series[name]
        .map((v, idx) => `${xOf(idx).toFixed(1)},${yOf(Number.isFinite(v) ? v : 0).toFixed(1)}`)
        .join(" ");
      return `<polyline points="${pts}" fill="none" stroke="${SERIES_COLORS[i % SERIES_COLORS.length]}"
              stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>`;
    })
    .join("");

  const legend = names
    .map(
      (name, i) =>
        `<span><i style="background:${SERIES_COLORS[i % SERIES_COLORS.length]}"></i>${name}</span>`
    )
    .join("");

  return `<div class="chart"><h3>${title}</h3>
    <svg viewBox="0 0 ${W} ${H}">${grid}${paths}</svg>
    <div class="chart-legend">${legend}</div></div>`;
}

function renderMetrics() {
  const pane = $("#tab-metrics");
  const m = state.metrics;
  if (!m) {
    pane.replaceChildren(el("p", { class: "empty" }, "No data yet."));
    return;
  }
  const services = Object.keys(m.metrics);
  const windows = m.metrics[services[0]]?.window_index || [];
  const pick = (metric) =>
    Object.fromEntries(services.map((s) => [s, m.metrics[s][metric] || []]));

  const charts = [
    lineChart("Request latency p95 — the symptom propagating", windows, pick("latency_p95"), (v) => Math.round(v) + "ms"),
    lineChart("Own processing latency p95 — where time is actually spent", windows, pick("self_latency_p95"), (v) => Math.round(v) + "ms"),
    lineChart("Error rate", windows, pick("error_rate"), (v) => (v * 100).toFixed(1) + "%"),
    lineChart("Anomaly score per service", m.scores.window_index, m.scores.series, (v) => v.toFixed(2)),
  ].join("");
  pane.innerHTML = `<div class="chart-grid">${charts}</div>
    <p class="hint" style="margin-top:12px">All panels share one time axis. The gap between the first
    curve to move here and the last is the propagation delay the ranker uses as evidence.</p>`;
}

/* ------------------------------------------------------------------- traces */
function renderTraces() {
  const pane = $("#tab-traces");
  const traces = state.incident?.traces || [];
  if (!traces.length) {
    pane.replaceChildren(el("p", { class: "empty" }, "No sampled traces for this incident yet."));
    return;
  }
  pane.innerHTML = traces
    .map((t) => {
      const t0 = Math.min(...t.spans.map((s) => s.start_time));
      const total = t.duration_ms || 1;
      const rows = t.spans
        .slice()
        .sort((a, b) => a.start_time - b.start_time)
        .map((s) => {
          const left = ((s.start_time - t0) * 1000 / total) * 100;
          const width = Math.max(0.6, (s.duration_ms / total) * 100);
          const cls =
            s.status !== "OK" ? "error" : s.service === t.dominant_service && s.kind === "server" ? "dominant" : "";
          return `<div class="span-row">
            <span class="span-name">${s.kind === "client" ? "↳ " : ""}${s.service} · ${s.operation}</span>
            <span class="span-track"><i class="span-bar ${cls}" style="left:${left.toFixed(1)}%;width:${Math.min(100 - left, width).toFixed(1)}%"></i></span>
            <span class="span-dur">${fmtMs(s.duration_ms)}</span>
          </div>`;
        })
        .join("");
      return `<div class="trace">
        <div class="trace-head">
          <span><code>${t.trace_id}</code> · total ${fmtMs(t.duration_ms)}${t.failed ? " · <b style='color:var(--critical)'>failed</b>" : ""}</span>
          <span>dominant self time: <b style="color:var(--cause)">${t.dominant_service}</b></span>
        </div>${rows}</div>`;
    })
    .join("");
}

/* --------------------------------------------------------------- candidates */
function renderCandidates() {
  const pane = $("#tab-candidates");
  const d = state.incident?.diagnosis;
  if (!d || !d.candidates.length) {
    pane.replaceChildren(el("p", { class: "empty" }, "No diagnosis yet."));
    return;
  }
  const featureNames = Object.keys(d.candidates[0].features);
  const head = `<tr><th>#</th><th>candidate</th><th class="num">score</th><th class="num">confidence</th>${featureNames
    .map((f) => `<th class="num">${f.replace(/_/g, " ")}</th>`)
    .join("")}</tr>`;
  const rows = d.candidates
    .map(
      (c) => `<tr class="${c.rank === 1 ? "is-cause" : ""}">
        <td class="num">${c.rank}</td>
        <td>${c.display_name}</td>
        <td class="num">${c.score.toFixed(3)}</td>
        <td class="num">${(c.confidence * 100).toFixed(0)}%</td>
        ${featureNames.map((f) => `<td class="num">${c.features[f].toFixed(2)}</td>`).join("")}
      </tr>`
    )
    .join("");

  const top = d.candidates[0];
  const contribs = Object.entries(top.contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxAbs = Math.max(1e-9, ...contribs.map(([, v]) => Math.abs(v)));
  const contribRows = contribs
    .map(
      ([k, v]) => `<tr><td>${k.replace(/_/g, " ")}</td>
        <td><div class="bar-cell"><div class="track"><i class="${v < 0 ? "neg" : ""}"
          style="width:${((Math.abs(v) / maxAbs) * 100).toFixed(0)}%"></i></div></div></td>
        <td class="num">${v >= 0 ? "+" : ""}${v.toFixed(3)}</td></tr>`
    )
    .join("");

  pane.innerHTML = `
    <p class="hint">Ranked by <code>${d.ranker}</code>. Every candidate is scored on the same features,
    so the ranking is inspectable rather than asserted.</p>
    <div style="overflow-x:auto"><table>${head}${rows}</table></div>
    <div class="subhead">Why ${top.display_name} scored highest</div>
    <table>${contribRows}</table>`;
}

/* --------------------------------------------------------------- evaluation */
async function loadEvaluation() {
  const pane = $("#tab-evaluation");
  try {
    state.evaluation = await api("/api/evaluation");
  } catch {
    state.evaluation = null;
    pane.innerHTML = `<p class="empty">No evaluation report yet. Run
      <code>make dataset &amp;&amp; make train &amp;&amp; make evaluate</code>.</p>`;
    return;
  }
  const r = state.evaluation;
  const detRows = Object.entries(r.detection)
    .map(
      ([name, d]) => `<tr><td><code>${name}</code></td>
        <td class="num">${d.incident_precision.toFixed(3)}</td>
        <td class="num">${d.incident_recall.toFixed(3)}</td>
        <td class="num">${d.f1.toFixed(3)}</td>
        <td class="num">${d.median_detection_delay_seconds ?? "—"} s</td>
        <td class="num">${d.false_alerts_per_hour.toFixed(2)}</td></tr>`
    )
    .join("");
  const diagRows = Object.entries(r.diagnosis)
    .map(
      ([name, d]) => `<tr><td><code>${name}</code></td>
        <td class="num">${d.top1_accuracy.toFixed(3)}</td>
        <td class="num">${d.top3_accuracy.toFixed(3)}</td>
        <td class="num">${d.mrr.toFixed(3)}</td></tr>`
    )
    .join("");
  const s = r.system || {};
  pane.innerHTML = `
    <p class="hint">Held-out test split: ${r.runs} runs, ${r.injected_incidents} injected incidents.
    Runs in this split were never used to fit a detector or a ranker.</p>
    <div class="subhead">Detection</div>
    <table><tr><th>detector</th><th class="num">precision</th><th class="num">recall</th>
      <th class="num">F1</th><th class="num">median delay</th><th class="num">false alerts/h</th></tr>${detRows}</table>
    <div class="subhead">Diagnosis (identical incidents for every ranker)</div>
    <table><tr><th>ranker</th><th class="num">top-1</th><th class="num">top-3</th><th class="num">MRR</th></tr>${diagRows}</table>
    <div class="subhead">System</div>
    <table>
      <tr><td>telemetry throughput</td><td class="num">${Math.round(s.telemetry_events_per_second || 0).toLocaleString()} events/s</td></tr>
      <tr><td>p95 window → diagnosis</td><td class="num">${(s.p95_window_to_diagnosis_ms || 0).toFixed(1)} ms</td></tr>
    </table>`;
}

/* ------------------------------------------------------------ incident list */
function renderIncidentList() {
  const box = $("#incident-list");
  if (!state.incidents.length) {
    box.replaceChildren(el("p", { class: "empty" }, "None yet."));
    return;
  }
  const table = el("table");
  table.append(
    el(
      "tr",
      {},
      ...["id", "status", "severity", "started", "detected after", "root cause", "confidence", "affected"].map((h) =>
        el("th", {}, h)
      )
    )
  );
  for (const inc of state.incidents) {
    const d = inc.diagnosis;
    const row = el(
      "tr",
      {
        class: "incident-row" + (inc.incident_id === state.selectedIncidentId ? " selected" : ""),
        onclick: async () => {
          state.selectedIncidentId = inc.incident_id;
          state.followLive = inc.status === "open";
          renderIncidentList();
          await refreshDetail();
        },
      },
      el("td", {}, el("code", {}, inc.incident_id)),
      el("td", {}, inc.status),
      el("td", {}, el("span", { class: "sev " + inc.severity_label }, inc.severity_label)),
      el("td", {}, inc.started_clock),
      el("td", { class: "num" }, inc.detection_delay_seconds + " s"),
      el("td", {}, d ? d.root_cause_display : "—"),
      el("td", { class: "num" }, d ? (d.confidence * 100).toFixed(0) + "%" : "—"),
      el("td", {}, inc.affected_services.join(", "))
    );
    table.append(row);
  }
  box.replaceChildren(table);
}

/* --------------------------------------------------------------------- boot */
(async function main() {
  await initControls();
  await refreshState();
  await refreshDetail();
  setInterval(() => refreshState().catch(console.error), POLL_STATE_MS);
  setInterval(() => refreshDetail().catch(console.error), POLL_DETAIL_MS);
})();
