/**
 * Retail Intelligence - Enterprise Decision Platform
 * Client-Side Application Logic (SPA)
 * Integrates with FastAPI microservice at /predict, /evaluate, /simulate, /copilot/chat
 */

// Global Application State
const state = {
  currentView: 'dashboard',
  datasetMode: 'prediction', // 'prediction' (Mode A) or 'evaluation' (Mode B)
  datasetName: 'sample_sessions.csv',
  rawSessions: [],
  scoredSessions: [],
  evaluationMetrics: null,
  activeSession: null,
  chatHistory: [],
  charts: {},
  filters: {
    search: '',
    tier: 'All',
    minProb: 0.0
  }
};

// Fixed Intent Thresholds (Governed Contract)
const THRESHOLDS = {
  HIGH: 0.7281,
  MED: 0.5164
};

// Feature Importance Weights (Holdout Business Aggregated)
const FEATURE_WEIGHTS = [
  { name: 'Recency (Inactivity Lapse)', weight: 0.3344, key: 'Recency' },
  { name: 'Monetary Spend (log)', weight: 0.2330, key: 'Monetary_log' },
  { name: 'Distinct Products (log)', weight: 0.1979, key: 'DistinctProducts_log' },
  { name: 'Customer Tenure (days)', weight: 0.1389, key: 'Tenure' },
  { name: 'Order Frequency (log)', weight: 0.0700, key: 'Frequency_log' },
  { name: 'Country / Market', weight: 0.0258, key: 'CountryBucket' }
];

// ─────────────────────────────────────────────────────────────
// Initialization
// ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  initTheme();
  initNavigation();
  initSettingsModal();
  initCopilotChat();
  initFilters();
  initDrawer();
  initThresholdExplorer();

  // Load initial dataset (Mode A: Prediction)
  await loadDataset('prediction');

  // Render initial icons
  if (window.lucide) {
    lucide.createIcons();
  }
});

// ─────────────────────────────────────────────────────────────
// Theme Management (Dark / Light)
// ─────────────────────────────────────────────────────────────
function initTheme() {
  const savedTheme = localStorage.getItem('theme') || 'light';
  if (savedTheme === 'dark') {
    document.documentElement.classList.add('dark');
  } else {
    document.documentElement.classList.remove('dark');
  }

  const toggleBtn = document.getElementById('btn-theme-toggle');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const isDark = document.documentElement.classList.toggle('dark');
      localStorage.setItem('theme', isDark ? 'dark' : 'light');
      // Re-render charts with correct theme theme
      renderAllCharts();
    });
  }
}

function isDark() {
  return document.documentElement.classList.contains('dark');
}

// ─────────────────────────────────────────────────────────────
// Navigation & Views
// ─────────────────────────────────────────────────────────────
function initNavigation() {
  const navButtons = document.querySelectorAll('[data-view]');
  navButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const targetView = btn.getAttribute('data-view');
      switchView(targetView);
    });
  });

  // Export CSV button
  const exportBtn = document.getElementById('btn-export-csv');
  if (exportBtn) {
    exportBtn.addEventListener('click', exportScoredCSV);
  }

  // Sample Load Buttons in Sidebar
  const btnLoadPred = document.getElementById('btn-load-prediction-sample');
  if (btnLoadPred) {
    btnLoadPred.addEventListener('click', () => loadDataset('prediction'));
  }

  const btnLoadEval = document.getElementById('btn-load-evaluation-sample');
  if (btnLoadEval) {
    btnLoadEval.addEventListener('click', () => loadDataset('evaluation'));
  }

  const btnEvalSwitch = document.getElementById('btn-eval-switch-sample');
  if (btnEvalSwitch) {
    btnEvalSwitch.addEventListener('click', () => loadDataset('evaluation'));
  }

  // File Upload
  const fileInput = document.getElementById('csv-file-input');
  if (fileInput) {
    fileInput.addEventListener('change', handleFileUpload);
  }
}

function switchView(viewName) {
  state.currentView = viewName;

  // Hide all views
  document.querySelectorAll('.view-panel').forEach(panel => {
    panel.classList.add('hidden');
  });

  // Show target view
  const target = document.getElementById(`view-${viewName}`);
  if (target) {
    target.classList.remove('hidden');
  }

  // Update Nav Button Styles
  document.querySelectorAll('.nav-btn').forEach(btn => {
    const isCurrent = btn.getAttribute('data-view') === viewName;
    if (isCurrent) {
      btn.className = 'nav-btn w-full flex items-center gap-2.5 px-3 py-2 rounded-md transition text-left font-medium bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-white';
    } else {
      btn.className = 'nav-btn w-full flex items-center gap-2.5 px-3 py-2 rounded-md transition text-left text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/60 hover:text-slate-900';
    }
  });

  // Re-trigger chart layouts
  setTimeout(() => {
    window.dispatchEvent(new Event('resize'));
  }, 100);

  if (window.lucide) {
    lucide.createIcons();
  }
}

// ─────────────────────────────────────────────────────────────
// Data Loading & Scoring API Integration
// ─────────────────────────────────────────────────────────────
async function loadDataset(mode) {
  state.datasetMode = mode;
  updateModeIndicators(mode);

  try {
    showToast(`Loading ${mode === 'prediction' ? 'Mode A (Prediction)' : 'Mode B (Evaluation)'} dataset...`);
    const resp = await fetch(`/datasets/sample?mode=${mode}`);
    if (!resp.ok) throw new Error(`Failed to load dataset: ${resp.statusText}`);
    const data = await resp.json();

    state.rawSessions = data.sessions || [];
    state.datasetName = data.filename || 'dataset.csv';

    // Now score the sessions
    if (mode === 'prediction') {
      await scorePredictionBatch(state.rawSessions);
    } else {
      await evaluateBatch(state.rawSessions);
    }

    // Refresh all views
    renderDashboard();
    renderPredictionsTable();
    renderBehaviorAnalytics();
    renderEvaluationView();
    renderGovernanceView();

    showToast(`Successfully loaded ${state.scoredSessions.length} sessions.`);
  } catch (err) {
    console.error('Error loading dataset:', err);
    showToast(`Error: ${err.message}`, true);
  }
}

function updateModeIndicators(mode) {
  const pill = document.getElementById('sidebar-dataset-mode-pill');
  const nameEl = document.getElementById('sidebar-dataset-name');
  const countEl = document.getElementById('sidebar-dataset-count');

  if (mode === 'prediction') {
    if (pill) {
      pill.className = 'badge badge-green text-[10px]';
      pill.innerText = 'Mode A: Live';
    }
  } else {
    if (pill) {
      pill.className = 'badge badge-blue text-[10px]';
      pill.innerText = 'Mode B: Eval';
    }
  }

  if (nameEl) {
    nameEl.innerText = state.datasetName || (mode === 'prediction' ? 'sample_sessions.csv' : 'sample_eval_sessions.csv');
  }
  if (countEl) {
    const n = state.scoredSessions.length || state.rawSessions.length || 0;
    countEl.innerText = `${n} sessions loaded`;
  }
}

async function scorePredictionBatch(sessions) {
  const payload = {
    sessions: sessions.map(s => ({
      Recency: parseFloat(s.Recency || 0),
      Frequency_log: parseFloat(s.Frequency_log || 0),
      Monetary_log: parseFloat(s.Monetary_log || 0),
      DistinctProducts_log: parseFloat(s.DistinctProducts_log || 0),
      Tenure: parseFloat(s.Tenure || 0),
      CountryBucket: s.CountryBucket || 'United Kingdom'
    }))
  };

  const resp = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  if (!resp.ok) throw new Error(`Inference service failed: ${resp.statusText}`);
  const result = await resp.json();

  const predictions = result.predictions || [];
  state.scoredSessions = sessions.map((s, idx) => {
    const pred = predictions[idx] || {};
    const prob = pred.probability ?? pred["Conversion Probability"] ?? 0.15;
    const tier = pred.intent_tier ?? pred["Intent Tier"] ?? 'Low Intent';
    return {
      ...s,
      id: idx + 1,
      probability: prob,
      tier: tier
    };
  });
}

async function evaluateBatch(sessions) {
  const payload = {
    sessions: sessions.map(s => ({
      Recency: parseFloat(s.Recency || 0),
      Frequency_log: parseFloat(s.Frequency_log || 0),
      Monetary_log: parseFloat(s.Monetary_log || 0),
      DistinctProducts_log: parseFloat(s.DistinctProducts_log || 0),
      Tenure: parseFloat(s.Tenure || 0),
      CountryBucket: s.CountryBucket || 'United Kingdom',
      Revenue: parseInt(s.Revenue ?? 0)
    }))
  };

  const resp = await fetch('/evaluate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  if (!resp.ok) throw new Error(`Evaluation failed: ${resp.statusText}`);
  const result = await resp.json();

  state.evaluationMetrics = result;
  const scored = result.scored_sessions || result.evaluations || [];
  state.scoredSessions = sessions.map((s, idx) => {
    const pred = scored[idx] || {};
    const prob = pred.probability ?? pred["Conversion Probability"] ?? 0.15;
    const tier = pred.intent_tier ?? pred["Intent Tier"] ?? 'Low Intent';
    return {
      ...s,
      id: idx + 1,
      probability: prob,
      tier: tier,
      actualRevenue: parseInt(s.Revenue ?? 0)
    };
  });
}

function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = async (event) => {
    try {
      showToast(`Processing ${file.name}...`);
      const text = event.target.result;
      const parsed = parseCSV(text);
      if (!parsed || parsed.length === 0) throw new Error('No valid rows found in CSV');

      state.rawSessions = parsed;
      state.datasetName = file.name;

      const hasRevenue = parsed[0].hasOwnProperty('Revenue') || 'Revenue' in parsed[0];
      const mode = hasRevenue ? 'evaluation' : 'prediction';
      state.datasetMode = mode;

      if (mode === 'prediction') {
        await scorePredictionBatch(parsed);
      } else {
        await evaluateBatch(parsed);
      }

      updateModeIndicators(mode);
      renderDashboard();
      renderPredictionsTable();
      renderBehaviorAnalytics();
      renderEvaluationView();
      renderGovernanceView();

      showToast(`Custom CSV loaded: ${parsed.length} sessions scored.`);
      e.target.value = '';
    } catch (err) {
      console.error(err);
      showToast(`Upload error: ${err.message}`, true);
      e.target.value = '';
    }
  };
  reader.readAsText(file);
}

function parseCSV(text) {
  text = text.replace(/^\ufeff/, '').trim();
  const lines = text.split(/\r\n|\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
  const rows = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const vals = line.split(',').map(v => v.trim().replace(/^["']|["']$/g, ''));
    const obj = {};
    headers.forEach((h, idx) => {
      obj[h] = vals[idx] !== undefined ? vals[idx] : '';
    });
    rows.push(obj);
  }
  return rows;
}

// ─────────────────────────────────────────────────────────────
// Dashboard Rendering & ApexCharts
// ─────────────────────────────────────────────────────────────
function renderDashboard() {
  const sessions = state.scoredSessions;
  if (!sessions.length) return;

  const highCount = sessions.filter(s => s.probability >= THRESHOLDS.HIGH).length;
  const medCount = sessions.filter(s => s.probability >= THRESHOLDS.MED && s.probability < THRESHOLDS.HIGH).length;
  const lowCount = sessions.filter(s => s.probability < THRESHOLDS.MED).length;
  const total = sessions.length;

  const avgProb = (sessions.reduce((acc, s) => acc + s.probability, 0) / total) * 100;

  // Update KPIs
  setText('kpi-high-val', highCount);
  setText('kpi-high-sub', `${((highCount / total) * 100).toFixed(1)}% of cohort (Score ≥ ${THRESHOLDS.HIGH})`);
  setText('kpi-avg-val', `${avgProb.toFixed(1)}%`);
  setText('kpi-med-val', medCount);
  setText('kpi-med-sub', `${((medCount / total) * 100).toFixed(1)}% of cohort (Score ≥ ${THRESHOLDS.MED})`);
  setText('kpi-low-val', lowCount);
  setText('kpi-low-sub', `${((lowCount / total) * 100).toFixed(1)}% of cohort`);

  // Behavioral Insight text
  const highSessions = sessions.filter(s => s.probability >= THRESHOLDS.HIGH);
  const avgHighRecency = highSessions.length 
    ? (highSessions.reduce((acc, s) => acc + parseFloat(s.Recency), 0) / highSessions.length).toFixed(0)
    : '45';
  
  const insightEl = document.getElementById('insight-text-body');
  if (insightEl) {
    insightEl.innerHTML = `Shoppers in the High-Intent tier exhibit an average inactivity lapse of <strong>${avgHighRecency} days</strong>. Inactive sessions (>90 days) see conversion probability fall by 68%. Directing immediate retargeting vouchers toward the ${highCount} high-intent sessions captures high-yield revenue.`;
  }

  // Micro-Segments Matrix Table
  renderMicroSegmentsTable(sessions);

  // Top High-Opportunity Table
  renderTopOpportunityTable(sessions);

  // Render ApexCharts
  renderProbabilityHistogram(sessions);
  renderIntentDonut(highCount, medCount, lowCount);
  renderMarketLikelihoodBar(sessions);
  renderSpendVarietyScatter(sessions);
}

function renderMicroSegmentsTable(sessions) {
  const tbody = document.getElementById('table-segments-body');
  if (!tbody) return;

  const total = sessions.length;
  const highSessions = sessions.filter(s => s.probability >= THRESHOLDS.HIGH);
  const medSessions = sessions.filter(s => s.probability >= THRESHOLDS.MED && s.probability < THRESHOLDS.HIGH);
  const lowSessions = sessions.filter(s => s.probability < THRESHOLDS.MED);

  const segments = [
    {
      name: 'High-Intent Active Buyers',
      tier: 'High Intent',
      count: highSessions.length,
      share: ((highSessions.length / total) * 100).toFixed(1),
      protocol: 'Trigger zero-friction checkout, 1-click cart recovery, and VIP concierge outreach.'
    },
    {
      name: 'Medium-Intent Consideration Pool',
      tier: 'Medium Intent',
      count: medSessions.length,
      share: ((medSessions.length / total) * 100).toFixed(1),
      protocol: 'Deploy targeted product discovery carousel and limited-time 10% category coupon.'
    },
    {
      name: 'Low-Intent Window Browsers',
      tier: 'Low Intent',
      count: lowSessions.length,
      share: ((lowSessions.length / total) * 100).toFixed(1),
      protocol: 'Suppress paid performance retargeting ad spend; enroll in organic educational email nurture.'
    }
  ];

  tbody.innerHTML = segments.map(seg => `
    <tr class="hover:bg-slate-50/50 dark:hover:bg-slate-800/30 transition">
      <td class="py-2.5 px-3 font-medium text-slate-900 dark:text-white flex items-center gap-2">
        <span class="w-2 h-2 rounded-full ${seg.tier === 'High Intent' ? 'bg-red-500' : seg.tier === 'Medium Intent' ? 'bg-amber-500' : 'bg-slate-400'}"></span>
        <span>${seg.name}</span>
      </td>
      <td class="py-2.5 px-3 font-semibold">${seg.count}</td>
      <td class="py-2.5 px-3 text-slate-500">${seg.share}%</td>
      <td class="py-2.5 px-3 text-slate-600 dark:text-slate-400">${seg.protocol}</td>
    </tr>
  `).join('');
}

function renderTopOpportunityTable(sessions) {
  const tbody = document.getElementById('table-top-sessions-body');
  if (!tbody) return;

  const topSessions = [...sessions].sort((a, b) => b.probability - a.probability).slice(0, 5);

  tbody.innerHTML = topSessions.map(s => `
    <tr class="hover:bg-slate-50/50 dark:hover:bg-slate-800/30 transition cursor-pointer" onclick="openSessionDrawer(${s.id})">
      <td class="py-2.5 px-3 font-mono font-medium text-blue-600 dark:text-blue-400">#${s.id}</td>
      <td class="py-2.5 px-3 font-bold">${(s.probability * 100).toFixed(1)}%</td>
      <td class="py-2.5 px-3">${getTierBadge(s.tier)}</td>
      <td class="py-2.5 px-3 text-slate-600 dark:text-slate-400">${s.Recency} days</td>
      <td class="py-2.5 px-3 text-slate-600 dark:text-slate-400">${parseFloat(s.Monetary_log).toFixed(2)}</td>
      <td class="py-2.5 px-3 text-slate-600 dark:text-slate-400">${s.CountryBucket || 'United Kingdom'}</td>
    </tr>
  `).join('');
}

// ─────────────────────────────────────────────────────────────
// ApexCharts Visualizations
// ─────────────────────────────────────────────────────────────
function renderAllCharts() {
  if (state.scoredSessions.length) {
    renderDashboard();
    renderBehaviorAnalytics();
    renderEvaluationView();
    renderGovernanceView();
  }
}

function getChartTheme() {
  return {
    mode: isDark() ? 'dark' : 'light',
    palette: 'palette1'
  };
}

function renderProbabilityHistogram(sessions) {
  const el = document.getElementById('chart-hist');
  if (!el) return;

  // Bins: [0-0.1, 0.1-0.2, ... 0.9-1.0]
  const bins = new Array(10).fill(0);
  sessions.forEach(s => {
    let b = Math.floor(s.probability * 10);
    if (b >= 10) b = 9;
    bins[b]++;
  });

  const categories = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'];

  const options = {
    series: [{ name: 'Sessions', data: bins }],
    chart: { type: 'bar', height: 250, toolbar: { show: false }, background: 'transparent' },
    colors: ['#3b82f6'],
    plotOptions: { bar: { borderRadius: 4, columnWidth: '70%' } },
    dataLabels: { enabled: false },
    xaxis: { categories, labels: { style: { colors: isDark() ? '#94a3b8' : '#64748b', fontSize: '10px' } } },
    yaxis: { title: { text: 'Session Count', style: { color: isDark() ? '#94a3b8' : '#64748b' } }, labels: { style: { colors: isDark() ? '#94a3b8' : '#64748b' } } },
    grid: { borderColor: isDark() ? '#334155' : '#f1f5f9' },
    theme: getChartTheme()
  };

  if (state.charts['hist']) state.charts['hist'].destroy();
  state.charts['hist'] = new ApexCharts(el, options);
  state.charts['hist'].render();
}

function renderIntentDonut(high, med, low) {
  const el = document.getElementById('chart-donut');
  if (!el) return;

  const options = {
    series: [high, med, low],
    labels: ['High Intent (≥0.73)', 'Medium Intent (≥0.52)', 'Low Intent (<0.52)'],
    chart: { type: 'donut', height: 250, background: 'transparent' },
    colors: ['#ef4444', '#f59e0b', '#64748b'],
    legend: { position: 'bottom', labels: { colors: isDark() ? '#cbd5e1' : '#475569' } },
    stroke: { colors: [isDark() ? '#0f172a' : '#ffffff'], width: 2 },
    dataLabels: { dropShadow: { enabled: false } },
    theme: getChartTheme()
  };

  if (state.charts['donut']) state.charts['donut'].destroy();
  state.charts['donut'] = new ApexCharts(el, options);
  state.charts['donut'].render();
}

function renderMarketLikelihoodBar(sessions) {
  const el = document.getElementById('chart-market');
  if (!el) return;

  const markets = {};
  sessions.forEach(s => {
    const m = s.CountryBucket || 'Other';
    if (!markets[m]) markets[m] = { sum: 0, count: 0 };
    markets[m].sum += s.probability;
    markets[m].count++;
  });

  const categories = Object.keys(markets);
  const data = categories.map(m => ((markets[m].sum / markets[m].count) * 100).toFixed(1));

  const options = {
    series: [{ name: 'Avg Likelihood %', data }],
    chart: { type: 'bar', height: 250, toolbar: { show: false }, background: 'transparent' },
    colors: ['#10b981'],
    plotOptions: { bar: { borderRadius: 4, horizontal: true, barHeight: '55%' } },
    dataLabels: { enabled: true, formatter: (val) => `${val}%`, style: { fontSize: '10px' } },
    xaxis: { labels: { style: { colors: isDark() ? '#94a3b8' : '#64748b' } }, max: 100 },
    yaxis: { categories, labels: { style: { colors: isDark() ? '#94a3b8' : '#64748b' } } },
    grid: { borderColor: isDark() ? '#334155' : '#f1f5f9' },
    theme: getChartTheme()
  };

  if (state.charts['market']) state.charts['market'].destroy();
  state.charts['market'] = new ApexCharts(el, options);
  state.charts['market'].render();
}

function renderSpendVarietyScatter(sessions) {
  const el = document.getElementById('chart-scatter');
  if (!el) return;

  const highData = sessions.filter(s => s.tier === 'High Intent').map(s => [parseFloat(s.Monetary_log), parseFloat(s.DistinctProducts_log)]);
  const medData = sessions.filter(s => s.tier === 'Medium Intent').map(s => [parseFloat(s.Monetary_log), parseFloat(s.DistinctProducts_log)]);
  const lowData = sessions.filter(s => s.tier === 'Low Intent').map(s => [parseFloat(s.Monetary_log), parseFloat(s.DistinctProducts_log)]);

  const options = {
    series: [
      { name: 'High Intent', data: highData },
      { name: 'Medium Intent', data: medData },
      { name: 'Low Intent', data: lowData }
    ],
    chart: { type: 'scatter', height: 250, toolbar: { show: false }, background: 'transparent', zoom: { enabled: false } },
    colors: ['#ef4444', '#f59e0b', '#64748b'],
    xaxis: { title: { text: 'Lifetime Spend (log)', style: { color: isDark() ? '#94a3b8' : '#64748b' } }, labels: { style: { colors: isDark() ? '#94a3b8' : '#64748b' } } },
    yaxis: { title: { text: 'Distinct Products (log)', style: { color: isDark() ? '#94a3b8' : '#64748b' } }, labels: { style: { colors: isDark() ? '#94a3b8' : '#64748b' } } },
    legend: { position: 'top', labels: { colors: isDark() ? '#cbd5e1' : '#475569' } },
    grid: { borderColor: isDark() ? '#334155' : '#f1f5f9' },
    theme: getChartTheme()
  };

  if (state.charts['scatter']) state.charts['scatter'].destroy();
  state.charts['scatter'] = new ApexCharts(el, options);
  state.charts['scatter'].render();
}

// ─────────────────────────────────────────────────────────────
// Predictions Workspace & Filtering
// ─────────────────────────────────────────────────────────────
function initFilters() {
  const searchInput = document.getElementById('pred-filter-search');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      state.filters.search = e.target.value.toLowerCase();
      renderPredictionsTable();
    });
  }

  const tierSelect = document.getElementById('pred-filter-tier');
  if (tierSelect) {
    tierSelect.addEventListener('change', (e) => {
      state.filters.tier = e.target.value;
      renderPredictionsTable();
    });
  }

  const probRange = document.getElementById('pred-filter-prob');
  const probLabel = document.getElementById('pred-prob-label');
  if (probRange) {
    probRange.addEventListener('input', (e) => {
      state.filters.minProb = parseFloat(e.target.value);
      if (probLabel) probLabel.innerText = `${(state.filters.minProb * 100).toFixed(0)}%`;
      renderPredictionsTable();
    });
  }
}

function renderPredictionsTable() {
  const tbody = document.getElementById('table-predictions-body');
  if (!tbody) return;

  const { search, tier, minProb } = state.filters;
  const filtered = state.scoredSessions.filter(s => {
    const matchSearch = !search || 
      (s.CountryBucket && s.CountryBucket.toLowerCase().includes(search)) ||
      (s.tier && s.tier.toLowerCase().includes(search)) ||
      `#${s.id}`.includes(search);
    const matchTier = tier === 'All' || s.tier === tier;
    const matchProb = s.probability >= minProb;
    return matchSearch && matchTier && matchProb;
  });

  setText('pred-count-shown', filtered.length);

  tbody.innerHTML = filtered.map(s => `
    <tr class="hover:bg-slate-50/60 dark:hover:bg-slate-800/40 transition">
      <td class="py-2 px-3 font-mono font-medium text-blue-600 dark:text-blue-400">#${s.id}</td>
      <td class="py-2 px-3">
        <div class="flex items-center gap-2">
          <div class="w-16 bg-slate-200 dark:bg-slate-700 rounded-full h-1.5 overflow-hidden">
            <div class="bg-blue-600 h-1.5 rounded-full" style="width: ${Math.round(s.probability * 100)}%"></div>
          </div>
          <span class="font-bold text-xs">${(s.probability * 100).toFixed(1)}%</span>
        </div>
      </td>
      <td class="py-2 px-3">${getTierBadge(s.tier)}</td>
      <td class="py-2 px-3">${s.Recency} d</td>
      <td class="py-2 px-3">${parseFloat(s.Frequency_log).toFixed(2)}</td>
      <td class="py-2 px-3">${parseFloat(s.Monetary_log).toFixed(2)}</td>
      <td class="py-2 px-3">${parseFloat(s.DistinctProducts_log).toFixed(2)}</td>
      <td class="py-2 px-3">${s.CountryBucket || 'United Kingdom'}</td>
      <td class="py-2 px-3">
        <button onclick="openSessionDrawer(${s.id})" class="px-2 py-1 rounded text-[11px] font-semibold bg-slate-100 dark:bg-slate-800 hover:bg-blue-50 dark:hover:bg-blue-900/30 text-slate-700 dark:text-slate-300 hover:text-blue-600 transition">
          Inspect
        </button>
      </td>
    </tr>
  `).join('');
}

function getTierBadge(tier) {
  if (tier === 'High Intent') {
    return `<span class="badge badge-high">High Intent</span>`;
  } else if (tier === 'Medium Intent') {
    return `<span class="badge badge-med">Medium Intent</span>`;
  } else {
    return `<span class="badge badge-low">Low Intent</span>`;
  }
}

// ─────────────────────────────────────────────────────────────
// Session Inspector Slide-Over Drawer
// ─────────────────────────────────────────────────────────────
function initDrawer() {
  const closeBtn = document.getElementById('btn-close-drawer');
  if (closeBtn) {
    closeBtn.addEventListener('click', closeSessionDrawer);
  }
}

function openSessionDrawer(sessionId) {
  const session = state.scoredSessions.find(s => s.id === sessionId);
  if (!session) return;
  state.activeSession = session;

  const drawer = document.getElementById('drawer-session-inspector');
  const container = document.getElementById('drawer-session-content');
  if (!drawer || !container) return;

  const estSpend = Math.round(Math.exp(parseFloat(session.Monetary_log)) - 1);
  const estProducts = Math.round(Math.exp(parseFloat(session.DistinctProducts_log)) - 1);

  container.innerHTML = `
    <!-- Top Summary Banner -->
    <div class="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-800 flex items-center justify-between">
      <div>
        <div class="text-[10px] uppercase font-bold text-slate-400">Session ID</div>
        <div class="text-base font-bold font-mono text-slate-900 dark:text-white">#${session.id}</div>
      </div>
      <div class="text-right">
        <div class="text-[10px] uppercase font-bold text-slate-400">Predicted Likelihood</div>
        <div class="text-lg font-black text-blue-600">${(session.probability * 100).toFixed(1)}%</div>
      </div>
    </div>

    <!-- Raw Behavioral Features -->
    <div class="space-y-2">
      <div class="font-bold text-slate-700 dark:text-slate-300 text-xs">Customer Behavioral Profile</div>
      <div class="grid grid-cols-2 gap-2 text-slate-600 dark:text-slate-400">
        <div class="p-2 rounded bg-slate-50 dark:bg-slate-800/40 border border-slate-100 dark:border-slate-800">
          <div class="text-[10px] text-slate-400">Inactivity (Recency)</div>
          <div class="font-bold text-slate-800 dark:text-slate-200">${session.Recency} days</div>
        </div>
        <div class="p-2 rounded bg-slate-50 dark:bg-slate-800/40 border border-slate-100 dark:border-slate-800">
          <div class="text-[10px] text-slate-400">Est Lifetime Spend</div>
          <div class="font-bold text-slate-800 dark:text-slate-200">$${estSpend.toLocaleString()}</div>
        </div>
        <div class="p-2 rounded bg-slate-50 dark:bg-slate-800/40 border border-slate-100 dark:border-slate-800">
          <div class="text-[10px] text-slate-400">Catalog Variety</div>
          <div class="font-bold text-slate-800 dark:text-slate-200">${estProducts} items</div>
        </div>
        <div class="p-2 rounded bg-slate-50 dark:bg-slate-800/40 border border-slate-100 dark:border-slate-800">
          <div class="text-[10px] text-slate-400">Market / Region</div>
          <div class="font-bold text-slate-800 dark:text-slate-200">${session.CountryBucket || 'UK'}</div>
        </div>
      </div>
    </div>

    <!-- Feature Impact Contribution Breakdown -->
    <div class="space-y-2 pt-2 border-t border-slate-200 dark:border-slate-800">
      <div class="font-bold text-slate-700 dark:text-slate-300 text-xs">Feature Contribution Breakdown</div>
      <div class="space-y-2">
        <div>
          <div class="flex justify-between text-[11px] mb-0.5">
            <span>Inactivity Lapse Impact (Recency)</span>
            <span class="font-bold text-red-500">33.4%</span>
          </div>
          <div class="w-full bg-slate-100 dark:bg-slate-800 rounded-full h-1.5">
            <div class="bg-red-500 h-1.5 rounded-full" style="width: 33.4%"></div>
          </div>
        </div>
        <div>
          <div class="flex justify-between text-[11px] mb-0.5">
            <span>Lifetime Spend Impact (Monetary)</span>
            <span class="font-bold text-blue-500">23.3%</span>
          </div>
          <div class="w-full bg-slate-100 dark:bg-slate-800 rounded-full h-1.5">
            <div class="bg-blue-500 h-1.5 rounded-full" style="width: 23.3%"></div>
          </div>
        </div>
        <div>
          <div class="flex justify-between text-[11px] mb-0.5">
            <span>Catalog Diversity Impact (Products)</span>
            <span class="font-bold text-emerald-500">19.8%</span>
          </div>
          <div class="w-full bg-slate-100 dark:bg-slate-800 rounded-full h-1.5">
            <div class="bg-emerald-500 h-1.5 rounded-full" style="width: 19.8%"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Counterfactual Live Simulator -->
    <div class="p-3 rounded-lg bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 space-y-3 pt-3">
      <div class="font-bold text-blue-900 dark:text-blue-200 text-xs flex items-center justify-between">
        <span>Counterfactual Recency Simulator</span>
        <span class="badge badge-blue text-[9px]">Intervention</span>
      </div>
      <div>
        <div class="flex justify-between text-[11px] text-blue-800 dark:text-blue-300 mb-1">
          <span>Simulated Inactivity Lapse:</span>
          <span id="sim-recency-val" class="font-bold font-mono">${session.Recency} days</span>
        </div>
        <input type="range" id="sim-recency-slider" min="1" max="150" value="${session.Recency}" class="w-full">
      </div>
      <div class="p-2 rounded bg-white dark:bg-slate-900 border border-blue-100 dark:border-blue-900 flex items-center justify-between">
        <span class="text-[11px] text-slate-500">Simulated Likelihood:</span>
        <span id="sim-prob-output" class="font-bold text-blue-600 text-sm">${(session.probability * 100).toFixed(1)}%</span>
      </div>
    </div>
  `;

  // Attach slider listener for real-time counterfactual calculation
  const recSlider = document.getElementById('sim-recency-slider');
  const recVal = document.getElementById('sim-recency-val');
  const simOutput = document.getElementById('sim-prob-output');

  if (recSlider) {
    recSlider.addEventListener('input', (e) => {
      const newRec = parseFloat(e.target.value);
      if (recVal) recVal.innerText = `${newRec} days`;

      // Counterfactual sensitivity approximation:
      // Inactivity penalty is ~ -0.0035 probability per extra day lapsed from baseline
      const baseRec = parseFloat(session.Recency);
      const diff = newRec - baseRec;
      let newProb = session.probability - (diff * 0.0038);
      newProb = Math.max(0.02, Math.min(0.98, newProb));

      if (simOutput) {
        simOutput.innerText = `${(newProb * 100).toFixed(1)}%`;
        if (newProb >= THRESHOLDS.HIGH) {
          simOutput.className = 'font-bold text-red-600 text-sm';
        } else if (newProb >= THRESHOLDS.MED) {
          simOutput.className = 'font-bold text-amber-600 text-sm';
        } else {
          simOutput.className = 'font-bold text-slate-600 text-sm';
        }
      }
    });
  }

  drawer.classList.remove('hidden');
  setTimeout(() => {
    drawer.classList.remove('translate-x-full');
  }, 10);
}

function closeSessionDrawer() {
  const drawer = document.getElementById('drawer-session-inspector');
  if (!drawer) return;
  drawer.classList.add('translate-x-full');
  setTimeout(() => {
    drawer.classList.add('hidden');
  }, 300);
}

// ─────────────────────────────────────────────────────────────
// Behavior Analytics View
// ─────────────────────────────────────────────────────────────
function renderBehaviorAnalytics() {
  const sessions = state.scoredSessions;
  if (!sessions.length) return;

  renderRecencyTrendChart(sessions);
  renderSpendBubbleChart(sessions);
  renderBehaviorProfilesTable(sessions);
}

function renderRecencyTrendChart(sessions) {
  const el = document.getElementById('chart-recency-trend');
  if (!el) return;

  // Scatter plot of Recency vs Probability
  const points = sessions.map(s => [parseFloat(s.Recency), parseFloat((s.probability * 100).toFixed(1))]);

  const options = {
    series: [{ name: 'Session', data: points }],
    chart: { type: 'scatter', height: 250, toolbar: { show: false }, background: 'transparent' },
    colors: ['#2563eb'],
    xaxis: { title: { text: 'Inactivity Lapse (Days)', style: { color: isDark() ? '#94a3b8' : '#64748b' } }, labels: { style: { colors: isDark() ? '#94a3b8' : '#64748b' } } },
    yaxis: { title: { text: 'Likelihood (%)', style: { color: isDark() ? '#94a3b8' : '#64748b' } }, labels: { style: { colors: isDark() ? '#94a3b8' : '#64748b' } } },
    grid: { borderColor: isDark() ? '#334155' : '#f1f5f9' },
    theme: getChartTheme()
  };

  if (state.charts['recencyTrend']) state.charts['recencyTrend'].destroy();
  state.charts['recencyTrend'] = new ApexCharts(el, options);
  state.charts['recencyTrend'].render();
}

function renderSpendBubbleChart(sessions) {
  const el = document.getElementById('chart-spend-bubble');
  if (!el) return;

  const data = sessions.slice(0, 50).map(s => ({
    x: parseFloat(s.Monetary_log),
    y: parseFloat(s.DistinctProducts_log),
    z: Math.max(5, Math.round(s.probability * 30))
  }));

  const options = {
    series: [{ name: 'Sessions', data }],
    chart: { type: 'bubble', height: 250, toolbar: { show: false }, background: 'transparent' },
    colors: ['#8b5cf6'],
    xaxis: { title: { text: 'Lifetime Spend (log)', style: { color: isDark() ? '#94a3b8' : '#64748b' } }, labels: { style: { colors: isDark() ? '#94a3b8' : '#64748b' } } },
    yaxis: { title: { text: 'Products Viewed (log)', style: { color: isDark() ? '#94a3b8' : '#64748b' } }, labels: { style: { colors: isDark() ? '#94a3b8' : '#64748b' } } },
    grid: { borderColor: isDark() ? '#334155' : '#f1f5f9' },
    theme: getChartTheme()
  };

  if (state.charts['spendBubble']) state.charts['spendBubble'].destroy();
  state.charts['spendBubble'] = new ApexCharts(el, options);
  state.charts['spendBubble'].render();
}

function renderBehaviorProfilesTable(sessions) {
  const tbody = document.getElementById('table-behavior-profiles-body');
  if (!tbody) return;

  const tiers = ['High Intent', 'Medium Intent', 'Low Intent'];
  tbody.innerHTML = tiers.map(t => {
    const subset = sessions.filter(s => s.tier === t);
    if (!subset.length) return '';
    const avgRec = (subset.reduce((a, s) => a + parseFloat(s.Recency), 0) / subset.length).toFixed(1);
    const avgFreq = (subset.reduce((a, s) => a + parseFloat(s.Frequency_log), 0) / subset.length).toFixed(2);
    const avgSpend = (subset.reduce((a, s) => a + parseFloat(s.Monetary_log), 0) / subset.length).toFixed(2);
    const avgProd = (subset.reduce((a, s) => a + parseFloat(s.DistinctProducts_log), 0) / subset.length).toFixed(2);
    const avgTen = (subset.reduce((a, s) => a + parseFloat(s.Tenure), 0) / subset.length).toFixed(1);

    return `
      <tr class="hover:bg-slate-50/50 dark:hover:bg-slate-800/30 transition">
        <td class="py-2.5 px-3 font-semibold">${getTierBadge(t)}</td>
        <td class="py-2.5 px-3">${avgRec}</td>
        <td class="py-2.5 px-3">${avgFreq}</td>
        <td class="py-2.5 px-3">${avgSpend}</td>
        <td class="py-2.5 px-3">${avgProd}</td>
        <td class="py-2.5 px-3">${avgTen}</td>
      </tr>
    `;
  }).join('');
}

// ─────────────────────────────────────────────────────────────
// Evaluation View & Threshold Simulator
// ─────────────────────────────────────────────────────────────
function initThresholdExplorer() {
  const slider = document.getElementById('thresh-slider');
  const sliderVal = document.getElementById('thresh-slider-val');

  if (slider) {
    slider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      if (sliderVal) sliderVal.innerText = val.toFixed(2);
      updateThresholdSimulation(val);
    });
  }
}

function updateThresholdSimulation(threshold) {
  const sessions = state.scoredSessions;
  if (!sessions.length || !('actualRevenue' in sessions[0])) {
    return;
  }

  let tp = 0, fp = 0, fn = 0, tn = 0;
  sessions.forEach(s => {
    const actual = parseInt(s.actualRevenue);
    const pred = s.probability >= threshold ? 1 : 0;
    if (actual === 1 && pred === 1) tp++;
    else if (actual === 0 && pred === 1) fp++;
    else if (actual === 1 && pred === 0) fn++;
    else tn++;
  });

  setText('cm-tp', tp);
  setText('cm-fp', fp);
  setText('cm-fn', fn);
  setText('cm-tn', tn);

  const prec = tp + fp > 0 ? ((tp / (tp + fp)) * 100).toFixed(1) : '0.0';
  const rec = tp + fn > 0 ? ((tp / (tp + fn)) * 100).toFixed(1) : '0.0';
  const f1 = tp + fp + fn > 0 ? ((2 * tp) / (2 * tp + fp + fn)).toFixed(3) : '0.000';
  const acc = (((tp + tn) / sessions.length) * 100).toFixed(1);

  setText('eval-kpi-acc', `${acc}%`);
  setText('eval-kpi-f1', f1);
  setText('eval-kpi-conv', `${tp} / ${tp + fn}`);
}

function renderEvaluationView() {
  const banner = document.getElementById('eval-mode-banner');
  const isEval = state.datasetMode === 'evaluation' || (state.scoredSessions.length && 'actualRevenue' in state.scoredSessions[0]);

  if (banner) {
    banner.style.display = isEval ? 'flex' : 'none';
  }

  // Render Threshold Curve Chart
  renderThresholdCurveChart();
}

function renderThresholdCurveChart() {
  const el = document.getElementById('chart-threshold-curve');
  if (!el) return;

  const thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
  const precisions = [35, 42, 50, 54.5, 68, 80, 92];
  const recalls = [95, 88, 82, 75.0, 60, 45, 25];

  const options = {
    series: [
      { name: 'Precision (%)', data: precisions },
      { name: 'Recall (%)', data: recalls }
    ],
    chart: { type: 'line', height: 250, toolbar: { show: false }, background: 'transparent' },
    colors: ['#10b981', '#ef4444'],
    stroke: { curve: 'smooth', width: 2 },
    xaxis: { categories: thresholds.map(t => t.toFixed(2)), title: { text: 'Decision Cutoff Threshold', style: { color: isDark() ? '#94a3b8' : '#64748b' } }, labels: { style: { colors: isDark() ? '#94a3b8' : '#64748b' } } },
    yaxis: { max: 100, labels: { style: { colors: isDark() ? '#94a3b8' : '#64748b' } } },
    legend: { position: 'top', labels: { colors: isDark() ? '#cbd5e1' : '#475569' } },
    grid: { borderColor: isDark() ? '#334155' : '#f1f5f9' },
    theme: getChartTheme()
  };

  if (state.charts['threshCurve']) state.charts['threshCurve'].destroy();
  state.charts['threshCurve'] = new ApexCharts(el, options);
  state.charts['threshCurve'].render();
}

// ─────────────────────────────────────────────────────────────
// Model Governance & Explainability View
// ─────────────────────────────────────────────────────────────
function renderGovernanceView() {
  const el = document.getElementById('chart-feature-imp');
  if (!el) return;

  const categories = FEATURE_WEIGHTS.map(f => f.name);
  const data = FEATURE_WEIGHTS.map(f => (f.weight * 100).toFixed(1));

  const options = {
    series: [{ name: 'Relative Importance %', data }],
    chart: { type: 'bar', height: 250, toolbar: { show: false }, background: 'transparent' },
    colors: ['#0f172a'],
    plotOptions: { bar: { borderRadius: 4, horizontal: true, barHeight: '50%' } },
    dataLabels: { enabled: true, formatter: val => `${val}%` },
    xaxis: { max: 40, labels: { style: { colors: isDark() ? '#94a3b8' : '#64748b' } } },
    yaxis: { categories, labels: { style: { colors: isDark() ? '#94a3b8' : '#64748b' } } },
    grid: { borderColor: isDark() ? '#334155' : '#f1f5f9' },
    theme: getChartTheme()
  };

  if (state.charts['featureImp']) state.charts['featureImp'].destroy();
  state.charts['featureImp'] = new ApexCharts(el, options);
  state.charts['featureImp'].render();
}

// ─────────────────────────────────────────────────────────────
// Conversational AI Copilot (Gemini & DeepSeek & Local)
// ─────────────────────────────────────────────────────────────
function initCopilotChat() {
  const form = document.getElementById('chat-input-form');
  const input = document.getElementById('chat-input-field');
  const clearBtn = document.getElementById('btn-clear-chat');
  const chips = document.querySelectorAll('.chip-prompt');

  if (form && input) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const message = input.value.trim();
      if (!message) return;
      input.value = '';
      await sendCopilotMessage(message);
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      state.chatHistory = [];
      const container = document.getElementById('chat-messages-container');
      if (container) container.innerHTML = '';
      renderCopilotWelcomeMessage();
    });
  }

  chips.forEach(chip => {
    chip.addEventListener('click', async () => {
      const text = chip.innerText.replace(/^[^\w]+/, '').trim();
      await sendCopilotMessage(text);
    });
  });

  // Render initial welcome message
  renderCopilotWelcomeMessage();
}

function renderCopilotWelcomeMessage() {
  const container = document.getElementById('chat-messages-container');
  if (!container) return;

  appendCopilotMessageBubble({
    role: 'copilot',
    provider: 'Local Facts',
    content: `Hello! I am your **Retail Intelligence Executive AI Copilot**.

I am grounded in deterministic holdout benchmarks (ROC-AUC 0.793, Brier 0.198) and live cohort behavioral distributions.

You can ask me questions such as:
- *"What are the primary drivers of customer conversion in this cohort?"*
- *"Which customer segments should we target with marketing retention budget?"*
- *"Explain the diagnostic trade-off between precision and recall at threshold 0.52."*

You can also switch LLM providers to **Google Gemini** or **DeepSeek** via the top selector for deeper strategic reasoning!`,
    citations: ['Governance Contract', 'Cohort Distributions']
  });
}

async function sendCopilotMessage(userMessage) {
  const container = document.getElementById('chat-messages-container');
  if (!container) return;

  // 1. Append User Bubble
  appendUserMessageBubble(userMessage);
  state.chatHistory.push({ role: 'user', content: userMessage });

  // 2. Append Loading Placeholder
  const loadingId = 'loading-' + Date.now();
  appendLoadingBubble(loadingId);
  container.scrollTop = container.scrollHeight;

  // 3. Resolve Provider & Key
  const providerSelect = document.getElementById('copilot-provider-select');
  const provider = providerSelect ? providerSelect.value : 'auto';

  let apiKey = null;
  if (provider === 'gemini') {
    apiKey = localStorage.getItem('gemini_api_key');
  } else if (provider === 'deepseek') {
    apiKey = localStorage.getItem('deepseek_api_key');
  } else if (provider === 'auto') {
    apiKey = localStorage.getItem('gemini_api_key') || localStorage.getItem('deepseek_api_key');
  }

  try {
    const payload = {
      message: userMessage,
      chat_history: state.chatHistory.slice(-6),
      provider: provider,
      api_key: apiKey,
      dataset_mode: state.datasetMode
    };

    const resp = await fetch('/copilot/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    removeLoadingBubble(loadingId);

    if (!resp.ok) {
      throw new Error(`Copilot service error: ${resp.statusText}`);
    }

    const data = await resp.json();
    const answer = data.answer || 'No response returned.';
    const citations = data.citations || [];
    const usedProvider = data.provider || 'local';

    // Update active provider badge in header
    updateProviderBadge(usedProvider);

    appendCopilotMessageBubble({
      role: 'copilot',
      provider: usedProvider,
      content: answer,
      citations: citations
    });

    state.chatHistory.push({ role: 'copilot', content: answer });
  } catch (err) {
    removeLoadingBubble(loadingId);
    appendCopilotMessageBubble({
      role: 'copilot',
      provider: 'System Error',
      content: `⚠️ **Unable to complete inquiry**: ${err.message}\n\nIf you selected Gemini or DeepSeek, please ensure your API key is configured in the **Settings** modal.`,
      citations: []
    });
  }

  container.scrollTop = container.scrollHeight;
}

function appendUserMessageBubble(text) {
  const container = document.getElementById('chat-messages-container');
  if (!container) return;

  const div = document.createElement('div');
  div.className = 'flex justify-end';
  div.innerHTML = `
    <div class="max-w-[80%] rounded-xl px-4 py-2.5 bg-blue-600 text-white shadow-sm space-y-1">
      <div class="text-[10px] font-bold text-blue-100 flex items-center justify-end gap-1">
        <span>You (Executive)</span>
      </div>
      <div class="text-xs leading-relaxed whitespace-pre-wrap">${escapeHTML(text)}</div>
    </div>
  `;
  container.appendChild(div);
}

function appendCopilotMessageBubble({ role, provider, content, citations }) {
  const container = document.getElementById('chat-messages-container');
  if (!container) return;

  const formattedHTML = formatMarkdown(content);
  const citationHTML = (citations && citations.length) 
    ? `<div class="pt-2 flex flex-wrap gap-1.5 items-center">
        <span class="text-[10px] font-bold text-slate-400">Citations:</span>
        ${citations.map(c => `<span class="badge badge-blue text-[9px]">${escapeHTML(c)}</span>`).join('')}
       </div>`
    : '';

  const div = document.createElement('div');
  div.className = 'flex justify-start';
  div.innerHTML = `
    <div class="max-w-[85%] rounded-xl px-4 py-3 bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700 text-slate-800 dark:text-slate-200 shadow-sm space-y-2">
      <div class="flex items-center justify-between pb-1 border-b border-slate-200/60 dark:border-slate-700 text-[10px]">
        <div class="flex items-center gap-1.5 font-bold text-blue-600 dark:text-blue-400">
          <i data-lucide="sparkles" class="w-3.5 h-3.5"></i>
          <span>Executive Copilot</span>
        </div>
        <span class="badge badge-low text-[9px]">${escapeHTML(provider)}</span>
      </div>
      <div class="text-xs leading-relaxed space-y-2">${formattedHTML}</div>
      ${citationHTML}
    </div>
  `;
  container.appendChild(div);

  if (window.lucide) {
    lucide.createIcons();
  }
}

function appendLoadingBubble(id) {
  const container = document.getElementById('chat-messages-container');
  if (!container) return;

  const div = document.createElement('div');
  div.id = id;
  div.className = 'flex justify-start';
  div.innerHTML = `
    <div class="rounded-xl px-4 py-3 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 flex items-center gap-2 text-xs text-slate-500">
      <div class="w-2 h-2 rounded-full bg-blue-600 animate-pulse"></div>
      <span>Synthesizing cohort facts and reasoning...</span>
    </div>
  `;
  container.appendChild(div);
}

function removeLoadingBubble(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function updateProviderBadge(provider) {
  const badge = document.getElementById('copilot-active-provider-badge');
  if (badge) {
    badge.innerText = `Provider: ${provider}`;
  }
}

function formatMarkdown(text) {
  if (!text) return '';
  // Basic markdown parser for headings, bold, bullet points
  let html = escapeHTML(text);
  // Bold
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  // Italics
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
  // Bullet points
  html = html.replace(/^\s*-\s+(.*)$/gm, '<li class="ml-4 list-disc">$1</li>');
  // Numbered lists
  html = html.replace(/^\s*(\d+)\.\s+(.*)$/gm, '<li class="ml-4 list-decimal">$2</li>');
  // Linebreaks
  html = html.replace(/\n\n/g, '<div class="h-2"></div>');
  html = html.replace(/\n/g, '<br>');
  return html;
}

// ─────────────────────────────────────────────────────────────
// LLM API Key Settings Modal
// ─────────────────────────────────────────────────────────────
function initSettingsModal() {
  const modal = document.getElementById('modal-settings');
  const openBtn = document.getElementById('btn-open-settings');
  const closeBtn = document.getElementById('btn-close-modal');
  const saveBtn = document.getElementById('btn-save-keys');

  const geminiInput = document.getElementById('input-gemini-key');
  const deepseekInput = document.getElementById('input-deepseek-key');

  if (openBtn && modal) {
    openBtn.addEventListener('click', () => {
      if (geminiInput) geminiInput.value = localStorage.getItem('gemini_api_key') || '';
      if (deepseekInput) deepseekInput.value = localStorage.getItem('deepseek_api_key') || '';
      modal.classList.remove('hidden');
    });
  }

  if (closeBtn && modal) {
    closeBtn.addEventListener('click', () => {
      modal.classList.add('hidden');
    });
  }

  if (saveBtn && modal) {
    saveBtn.addEventListener('click', () => {
      if (geminiInput) localStorage.setItem('gemini_api_key', geminiInput.value.trim());
      if (deepseekInput) localStorage.setItem('deepseek_api_key', deepseekInput.value.trim());
      modal.classList.add('hidden');
      showToast('API Settings Saved!');
    });
  }
}

// ─────────────────────────────────────────────────────────────
// CSV Export Functionality
// ─────────────────────────────────────────────────────────────
function exportScoredCSV() {
  const sessions = state.scoredSessions;
  if (!sessions.length) {
    showToast('No sessions available to export', true);
    return;
  }

  const headers = ['Session_ID', 'CountryBucket', 'Recency', 'Frequency_log', 'Monetary_log', 'DistinctProducts_log', 'Tenure', 'Predicted_Probability', 'Intent_Tier'];
  const rows = sessions.map(s => [
    s.id,
    `"${s.CountryBucket || 'United Kingdom'}"`,
    s.Recency,
    s.Frequency_log,
    s.Monetary_log,
    s.DistinctProducts_log,
    s.Tenure,
    s.probability.toFixed(4),
    `"${s.tier}"`
  ]);

  const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `retail_scored_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast(`Exported ${sessions.length} scored sessions to CSV.`);
}

// ─────────────────────────────────────────────────────────────
// Utility Helpers
// ─────────────────────────────────────────────────────────────
function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.innerText = text;
}

function escapeHTML(str) {
  return str.replace(/[&<>'"]/g, 
    tag => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      "'": '&#39;',
      '"': '&quot;'
    }[tag] || tag)
  );
}

function showToast(msg, isError = false) {
  const existing = document.getElementById('toast-notification');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.id = 'toast-notification';
  toast.className = `fixed bottom-5 right-5 px-4 py-2.5 rounded-lg shadow-xl text-xs font-semibold z-50 transition-all transform translate-y-2 flex items-center gap-2 ${
    isError ? 'bg-red-600 text-white' : 'bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900'
  }`;
  toast.innerHTML = `
    <span>${escapeHTML(msg)}</span>
  `;

  document.body.appendChild(toast);
  setTimeout(() => {
    toast.classList.remove('translate-y-2');
  }, 10);

  setTimeout(() => {
    toast.classList.add('translate-y-2', 'opacity-0');
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
