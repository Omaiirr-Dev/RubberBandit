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
      priceHistory.push(state.price);
      if (priceHistory.length > MAX_CHART_POINTS) priceHistory.shift();
      render(state);
    };
  }

  // ---- Demo mode ----

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
    }
  }

  // ---- Render ----

  function render(s) {
    // Demo indicator
    if (s.demo !== undefined) updateDemoUI(s.demo);

    // Price
    $("#price").textContent = `$${s.price.toFixed(2)}`;

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

    // Price line
    ctx.beginPath();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    for (let i = 0; i < prices.length; i++) {
      const x = (i / (prices.length - 1)) * w;
      const y = toY(prices[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Glow fill
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(255,255,255,0.08)");
    grad.addColorStop(1, "rgba(255,255,255,0)");
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

    // Also catch visibility change to immediately re-check timer
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && timerEndAt && !timerDone) {
        tickTimer();
      }
    });

    window.addEventListener("resize", drawChart);
  });
})();
