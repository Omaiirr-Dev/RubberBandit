// RubberBand Frontend
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  let ws;
  let state = {};
  let priceHistory = [];
  const MAX_CHART_POINTS = 120;

  // Timer state — uses absolute timestamps so it works when backgrounded
  let timerEndAt = null; // Date.now() ms when timer finishes
  let timerInterval = null;
  let timerDone = false;

  // ---- WebSocket ----

  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws`);

    ws.onopen = () => {
      $("#connection-dot").className = "dot connected";
      $("#connection-text").textContent = "LIVE";
    };

    ws.onclose = () => {
      $("#connection-dot").className = "dot disconnected";
      $("#connection-text").textContent = "OFFLINE";
      setTimeout(connect, 3000);
    };

    ws.onerror = () => ws.close();

    ws.onmessage = (e) => {
      state = JSON.parse(e.data);
      if (state.price > 0) {
        priceHistory.push(state.price);
        if (priceHistory.length > MAX_CHART_POINTS) priceHistory.shift();
      }
      render(state);
    };
  }

  // ---- Demo mode ----

  let wasDemo = false;

  function updateDemoUI(isDemo) {
    const banner = $("#demo-banner");
    const btn = $("#demo-btn");
    if (isDemo) {
      banner.classList.add("show");
      btn.classList.add("active");
      btn.textContent = "DEMO ON";
    } else {
      banner.classList.remove("show");
      btn.classList.remove("active");
      btn.textContent = "DEMO";
      // Clear stale demo data when switching off
      if (wasDemo) {
        priceHistory = [];
        const canvas = $("#chart");
        if (canvas) {
          const ctx = canvas.getContext("2d");
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
    }
    wasDemo = isDemo;
  }

  // ---- Context ----

  function updateContextUI(s) {
    const banner = $("#context-banner");
    const btn = $("#ctx-btn");
    const card = $("#context-card");

    if (s.context_active) {
      banner.classList.add("show");
      btn.classList.remove("loading");
      btn.classList.add("active");
      btn.textContent = "PULL";
      card.classList.add("show");
      // Update range display
      const range = s.context_high > 0 && s.context_low > 0
        ? `$${s.context_low.toFixed(2)} — $${s.context_high.toFixed(2)}`
        : "";
      $("#ctx-range").textContent = range;
      $("#ctx-high").textContent = s.context_high > 0 ? `$${s.context_high.toFixed(2)}` : "---";
      $("#ctx-low").textContent = s.context_low > 0 ? `$${s.context_low.toFixed(2)}` : "---";
      $("#ctx-vwap").textContent = s.context_vwap > 0 ? `$${s.context_vwap.toFixed(2)}` : "---";
    } else {
      banner.classList.remove("show");
      btn.classList.remove("active", "loading");
      btn.textContent = "PULL";
      card.classList.remove("show");
    }

    // Handle error messages
    if (s.context_error) {
      btn.classList.remove("loading");
      btn.textContent = "PULL";
    }
  }

  // ---- Render ----

  function render(s) {
    // Demo indicator
    if (s.demo !== undefined) updateDemoUI(s.demo);

    // Context indicator
    if (s.context_active !== undefined) updateContextUI(s);

    // Price
    $("#price").textContent = s.price > 0 ? `$${s.price.toFixed(2)}` : "---";

    // Levels
    $("#support").textContent = s.support_floor > 0 ? `$${s.support_floor.toFixed(2)}` : "---";
    $("#resistance").textContent = s.resistance_ceiling > 0 ? `$${s.resistance_ceiling.toFixed(2)}` : "---";
    $("#mode").textContent = s.mode_price > 0 ? `$${s.mode_price.toFixed(2)}` : "---";
    $("#vwap").textContent = s.vwap > 0 ? `$${s.vwap.toFixed(2)}` : "---";

    // Signal gauge
    const score = s.signal_score;
    $("#gauge-fill").style.width = `${score}%`;

    const gauge = $("#gauge-fill");
    gauge.className = "gauge-fill";
    if (score >= 70) gauge.classList.add("high");
    else if (score >= 40) gauge.classList.add("mid");
    else gauge.classList.add("low");

    // Action card
    const card = $("#action-card");
    $("#action-text").textContent = s.action;
    $("#signal-pct").textContent = `${score}%`;
    card.className = "action-card";
    if (s.action === "BUY") {
      card.classList.add("buy");
    } else if (s.action === "SELL") {
      card.classList.add("sell");
    } else {
      card.classList.add("wait");
    }

    // Window fill
    $("#window-fill").style.width = `${s.window_fill}%`;
    $("#tick-count").textContent = `${s.tick_count}`;

    // Chart
    drawChart();
  }

  // ---- Chart ----

  function drawChart() {
    const canvas = $("#chart");
    if (!canvas || priceHistory.length < 2) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    // Guard: skip draw if layout hasn't settled yet (0 dimensions), retry next frame
    if (rect.width < 10 || rect.height < 10) {
      requestAnimationFrame(drawChart);
      return;
    }

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const totalW = rect.width;
    const h = rect.height;
    const rm = 38; // right margin for labels
    const w = totalW - rm;

    ctx.clearRect(0, 0, totalW, h);

    const prices = priceHistory;
    const min = Math.min(...prices) - 0.05;
    const max = Math.max(...prices) + 0.05;
    const range = max - min || 1;
    const toY = (p) => h - ((p - min) / range) * h;

    // Current price Y for collision avoidance
    const curY = state.price > 0 ? toY(state.price) : -100;
    const usedLabelYs = [curY]; // reserve space for current price

    // Helper: draw level line + label (skip label if too close to another)
    function drawLevel(price, color, dash) {
      if (price <= 0) return;
      const y = toY(price);
      if (y < 0 || y > h) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
      ctx.setLineDash([]);
      // Only draw label if not overlapping another
      const tooClose = usedLabelYs.some(ly => Math.abs(ly - y) < 12);
      if (!tooClose) {
        ctx.font = "8px 'JetBrains Mono', monospace";
        ctx.fillStyle = color;
        ctx.textAlign = "left";
        ctx.fillText(price.toFixed(2), w + 3, y + 3);
        usedLabelYs.push(y);
      }
    }

    drawLevel(state.support_floor, "rgba(74,222,128,0.5)", [4, 4]);
    drawLevel(state.resistance_ceiling, "rgba(248,113,113,0.5)", [4, 4]);
    drawLevel(state.vwap, "rgba(96,165,250,0.5)", [2, 3]);

    // Context range lines (cyan, long dash)
    if (state.context_active) {
      drawLevel(state.context_high, "rgba(34,211,238,0.6)", [8, 4]);
      drawLevel(state.context_low, "rgba(34,211,238,0.6)", [8, 4]);
    }

    // High/low range labels (skip if overlapping)
    if (!usedLabelYs.some(ly => Math.abs(ly - 6) < 12)) {
      ctx.font = "7px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#444";
      ctx.textAlign = "left";
      ctx.fillText(max.toFixed(2), w + 3, 8);
    }
    if (!usedLabelYs.some(ly => Math.abs(ly - (h - 4)) < 12)) {
      ctx.font = "7px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#444";
      ctx.textAlign = "left";
      ctx.fillText(min.toFixed(2), w + 3, h - 2);
    }

    // Price line — green segments when going up, red when going down
    ctx.lineWidth = 1.5;
    for (let i = 1; i < prices.length; i++) {
      const x0 = ((i - 1) / (prices.length - 1)) * w;
      const y0 = toY(prices[i - 1]);
      const x1 = (i / (prices.length - 1)) * w;
      const y1 = toY(prices[i]);
      ctx.beginPath();
      ctx.strokeStyle = prices[i] >= prices[i - 1]
        ? "rgba(74,222,128,0.9)"
        : "rgba(248,113,113,0.9)";
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }

    // Green glow fill
    ctx.beginPath();
    for (let i = 0; i < prices.length; i++) {
      const x = (i / (prices.length - 1)) * w;
      const y = toY(prices[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(74,222,128,0.1)");
    grad.addColorStop(1, "rgba(74,222,128,0)");
    ctx.fillStyle = grad;
    ctx.fill();

    // Current price tag
    if (state.price > 0) {
      const cy = toY(state.price);
      ctx.fillStyle = "#f0f0f0";
      ctx.fillRect(w + 1, cy - 5, rm - 2, 11);
      ctx.font = "bold 7px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#0a0a0a";
      ctx.textAlign = "left";
      ctx.fillText(state.price.toFixed(2), w + 3, cy + 3);
    }
  }

  // ---- Timer ----
  // Uses Date.now() timestamps so the countdown stays accurate even if
  // the phone goes to the homescreen and iOS throttles setInterval.

  function startTimer() {
    const mins = parseInt($("#input-mins").value);
    if (!mins || mins < 1) return;

    // Reset engine for fresh evaluation (keeps context)
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ cmd: "reset_window" }));
    }
    // Clear frontend data
    priceHistory = [];
    const canvas = $("#chart");
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    timerDone = false;
    timerEndAt = Date.now() + mins * 60 * 1000;
    $("#input-mins").disabled = true;
    $("#btn-start").textContent = "...";
    $("#btn-start").disabled = true;

    // Tick every 250ms for snappy updates, even after waking from background
    timerInterval = setInterval(tickTimer, 250);
    tickTimer();
  }

  function tickTimer() {
    const display = $("#timer-display");
    const remaining = timerEndAt - Date.now();

    if (remaining <= 0) {
      // Timer done
      display.textContent = "DONE";
      display.className = "done";
      timerDone = true;
      clearInterval(timerInterval);
      timerInterval = null;
      $("#btn-start").textContent = "Start";
      $("#btn-start").disabled = false;
      return;
    }

    const totalSec = Math.ceil(remaining / 1000);
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    display.textContent = `${m}:${s.toString().padStart(2, "0")}`;
    display.className = "running";
  }

  function resetTimer() {
    // Stop timer
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
    timerEndAt = null;
    timerDone = false;

    // Reset display
    $("#timer-display").textContent = "--:--";
    $("#timer-display").className = "idle";
    $("#input-mins").value = "";
    $("#input-mins").disabled = false;
    $("#btn-start").textContent = "Start";
    $("#btn-start").disabled = false;

    // Clear all numbers back to defaults
    priceHistory = [];
    state = {};
    $("#price").textContent = "$0.00";
    $("#support").textContent = "---";
    $("#resistance").textContent = "---";
    $("#mode").textContent = "---";
    $("#vwap").textContent = "---";
    $("#gauge-fill").style.width = "0%";
    $("#gauge-fill").className = "gauge-fill low";
    $("#action-text").textContent = "WAIT";
    $("#signal-pct").textContent = "0%";
    $("#action-card").className = "action-card wait";
    $("#window-fill").style.width = "0%";
    $("#tick-count").textContent = "0";

    // Clear context display
    $("#context-banner").classList.remove("show");
    $("#context-card").classList.remove("show");
    $("#ctx-btn").classList.remove("active", "loading");
    $("#ctx-btn").textContent = "PULL";

    // Clear canvas
    const canvas = $("#chart");
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  // ---- Init ----

  document.addEventListener("DOMContentLoaded", () => {
    connect();

    $("#btn-start").addEventListener("click", startTimer);
    $("#btn-reset").addEventListener("click", resetTimer);

    $("#demo-btn").addEventListener("click", () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ cmd: "toggle_demo" }));
      }
    });

    // Context: unit toggle (M ↔ S)
    let ctxUnit = "M"; // M = minutes, S = seconds
    $("#ctx-unit").addEventListener("click", () => {
      ctxUnit = ctxUnit === "M" ? "S" : "M";
      $("#ctx-unit").textContent = ctxUnit;
    });

    // Context: pull button
    $("#ctx-btn").addEventListener("click", () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        const btn = $("#ctx-btn");
        const val = parseInt($("#ctx-input").value) || 30;
        const minutes = ctxUnit === "S" ? val / 60 : val;
        btn.classList.add("loading");
        btn.textContent = "...";
        ws.send(JSON.stringify({ cmd: "pull_context", minutes }));
      }
    });

    // Also catch visibility change to immediately re-check timer
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && timerEndAt && !timerDone) {
        tickTimer();
      }
    });

    // ResizeObserver redraws chart whenever its container changes size
    // (handles first-load layout settling + window resize)
    const chartWrap = $(".chart-wrap");
    if (chartWrap && typeof ResizeObserver !== "undefined") {
      new ResizeObserver(() => drawChart()).observe(chartWrap);
    } else {
      window.addEventListener("resize", drawChart);
    }
  });
})();
