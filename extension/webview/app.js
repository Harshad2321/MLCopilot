/* =============================================================
   MLCopilot – Frontend Application (Vanilla JS + Chart.js)
   ============================================================= */

// ─── DOM refs ──────────────────────────────────────────────────
const $badge       = document.getElementById("statusBadge");
const $mEpoch      = document.getElementById("mEpoch");
const $mBatch      = document.getElementById("mBatch");
const $mLoss       = document.getElementById("mLoss");
const $mGrad       = document.getElementById("mGrad");
const $mLR         = document.getElementById("mLR");
const $mStep       = document.getElementById("mStep");
const $detection   = document.getElementById("detectionContent");
const $recs        = document.getElementById("recsContent");
const $btnStart    = document.getElementById("btnStart");
const $btnStop     = document.getElementById("btnStop");
const $lrInput     = document.getElementById("lrInput");

// ─── State ─────────────────────────────────────────────────────
const MAX_POINTS = 300;
const lossData   = [];
const gradData   = [];
const labels     = [];
const anomalyPoints = { loss: [], grad: [] };
let ws = null;

// ─── Chart Setup ───────────────────────────────────────────────
const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    interaction: { mode: "nearest", intersect: false },
    plugins: {
        legend: { display: false },
        tooltip: { enabled: true, backgroundColor: "#161b22", titleColor: "#e6edf3", bodyColor: "#8b949e", borderColor: "#30363d", borderWidth: 1 }
    },
    scales: {
        x: {
            display: true,
            grid: { color: "rgba(48,54,61,0.4)" },
            ticks: { color: "#8b949e", font: { size: 10 }, maxTicksLimit: 10 }
        },
        y: {
            display: true,
            grid: { color: "rgba(48,54,61,0.4)" },
            ticks: { color: "#8b949e", font: { size: 10 } }
        }
    }
};

const lossChart = new Chart(document.getElementById("lossChart"), {
    type: "line",
    data: {
        labels: labels,
        datasets: [
            {
                label: "Loss",
                data: lossData,
                borderColor: "#58a6ff",
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
                fill: { target: "origin", above: "rgba(88,166,255,0.06)" }
            },
            {
                label: "Anomaly",
                data: [],
                borderColor: "transparent",
                backgroundColor: "#f85149",
                pointRadius: 5,
                pointStyle: "circle",
                showLine: false
            }
        ]
    },
    options: { ...chartDefaults }
});

const gradChart = new Chart(document.getElementById("gradChart"), {
    type: "line",
    data: {
        labels: labels,
        datasets: [
            {
                label: "Grad Norm",
                data: gradData,
                borderColor: "#bc8cff",
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
                fill: { target: "origin", above: "rgba(188,140,255,0.06)" }
            },
            {
                label: "Anomaly",
                data: [],
                borderColor: "transparent",
                backgroundColor: "#f85149",
                pointRadius: 5,
                pointStyle: "circle",
                showLine: false
            }
        ]
    },
    options: { ...chartDefaults }
});

// ─── WebSocket ─────────────────────────────────────────────────
function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);

    ws.onopen = () => console.log("[ws] connected");
    ws.onclose = () => { console.log("[ws] disconnected"); setTimeout(connect, 2000); };
    ws.onerror = (e) => console.warn("[ws] error", e);

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "metric")  handleMetric(msg);
        if (msg.type === "status")  handleStatus(msg);
        if (msg.type === "lr_changed") $lrInput.value = msg.lr;
    };
}

// ─── Metric Handler ────────────────────────────────────────────
function handleMetric(m) {
    const step = m.global_batch;

    // Push data
    labels.push(step);
    lossData.push(m.loss);
    gradData.push(m.grad_norm);

    // Trim if needed
    if (labels.length > MAX_POINTS) {
        labels.shift(); lossData.shift(); gradData.shift();
        // Also trim anomaly datasets
        trimAnomalyData(lossChart.data.datasets[1].data);
        trimAnomalyData(gradChart.data.datasets[1].data);
    }

    // Mark anomaly points
    if (m.issue) {
        lossChart.data.datasets[1].data.push({ x: step, y: m.loss });
        gradChart.data.datasets[1].data.push({ x: step, y: m.grad_norm });
    }

    // Update charts
    lossChart.update("none");
    gradChart.update("none");

    // Update metric panel
    $mEpoch.textContent = m.epoch;
    $mBatch.textContent = m.batch;
    $mLoss.textContent  = m.loss != null ? m.loss.toFixed(4) : "NaN";
    $mGrad.textContent  = m.grad_norm != null ? m.grad_norm.toFixed(4) : "NaN";
    $mLR.textContent    = m.lr.toExponential(2);
    $mStep.textContent  = step;

    // Highlight loss if bad
    $mLoss.style.color = (m.loss != null && m.loss > 10) ? "#f85149" : "";
    $mGrad.style.color = (m.grad_norm != null && m.grad_norm > 10) ? "#f85149" : "";

    // Detection & recommendation panels
    if (m.issue) {
        renderDetection(m.issue);
        renderRecommendations(m.issue.recommendations || []);
        updateBadgeSeverity(m.issue.severity);
    }
}

function trimAnomalyData(arr) {
    const minLabel = labels[0];
    while (arr.length > 0 && arr[0].x < minLabel) arr.shift();
}

// ─── Status Handler ────────────────────────────────────────────
function handleStatus(msg) {
    const s = msg.status;
    setBadge(s);
    $btnStart.disabled = (s === "running");
    $btnStop.disabled  = (s !== "running");

    if (s === "stopped" || s === "error") {
        $btnStart.disabled = false;
        $btnStop.disabled = true;
    }
}

function setBadge(status) {
    const map = {
        idle:    ["IDLE",     "badge-idle"],
        running: ["RUNNING",  "badge-running"],
        stopped: ["STOPPED",  "badge-stopped"],
        error:   ["ERROR",    "badge-error"]
    };
    const [text, cls] = map[status] || map.idle;
    $badge.textContent = text;
    $badge.className = "badge " + cls;
}

function updateBadgeSeverity(severity) {
    const map = {
        critical: ["CRITICAL", "badge-critical"],
        high:     ["WARNING",  "badge-warning"],
        medium:   ["WARNING",  "badge-warning"],
        low:      ["HEALTHY",  "badge-running"]
    };
    const [text, cls] = map[severity] || ["RUNNING", "badge-running"];
    $badge.textContent = text;
    $badge.className = "badge " + cls;
}

// ─── Render Helpers ────────────────────────────────────────────
function renderDetection(issue) {
    const sev = issue.severity || "medium";
    $detection.innerHTML = `
        <div class="detection-alert ${sev}">
            <div class="det-type">${formatAnomaly(issue.anomaly)} <span style="opacity:0.6">(${(issue.confidence * 100).toFixed(0)}% confidence)</span></div>
            <div class="det-desc">${issue.description}</div>
            <div class="det-cause">Root Cause: ${issue.cause}</div>
        </div>
        <div style="font-size:0.8rem; color:#8b949e; margin-top:6px; line-height:1.5;">${issue.reasoning}</div>
    `;
}

function renderRecommendations(recs) {
    if (!recs.length) { $recs.innerHTML = '<div class="placeholder-text">No recommendations</div>'; return; }
    $recs.innerHTML = recs.map(r => `
        <div class="rec-item">
            <div class="rec-header">
                <span class="rec-action">${r.action}</span>
                <span class="rec-priority ${r.priority}">${r.priority}</span>
            </div>
            <div class="rec-body">
                ${r.reasoning}
                ${r.code_example ? `<code>${escapeHtml(r.code_example)}</code>` : ""}
            </div>
        </div>
    `).join("");
}

function formatAnomaly(type) {
    return (type || "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
}

// ─── API Calls ─────────────────────────────────────────────────
async function startTraining() {
    // Reset charts
    labels.length = 0; lossData.length = 0; gradData.length = 0;
    lossChart.data.datasets[1].data = [];
    gradChart.data.datasets[1].data = [];
    lossChart.update("none"); gradChart.update("none");
    $detection.innerHTML = '<div class="placeholder-text">No issues detected</div>';
    $recs.innerHTML = '<div class="placeholder-text">Waiting for analysis…</div>';

    $btnStart.disabled = true;
    const res = await fetch("/start", { method: "POST" });
    const data = await res.json();
    if (!data.ok) { alert(data.message); $btnStart.disabled = false; }
}

async function stopTraining() {
    $btnStop.disabled = true;
    await fetch("/stop", { method: "POST" });
}

async function applyLR() {
    const lr = parseFloat($lrInput.value);
    if (isNaN(lr) || lr <= 0) { alert("Enter a valid positive learning rate"); return; }
    const res = await fetch("/set_lr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lr })
    });
    const data = await res.json();
    if (data.ok) $lrInput.value = data.lr;
}

// ─── Init ──────────────────────────────────────────────────────
connect();
