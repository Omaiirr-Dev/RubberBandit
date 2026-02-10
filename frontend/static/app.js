// RubberBand Frontend
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  let ws;
  let state = {};
  let priceHistory = [];
  const MAX_CHART_POINTS = 200;
  const CANDLE_GROUP = 4; // ticks per candle

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

  // ---- Render ----

  function render(s) {
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

  function buildCandles(prices) {
    const candles = [];
    for (let i = 0; i < prices.length; i += CANDLE_GROUP) {
      const slice = prices.slice(i, i + CANDLE_GROUP);
      candles.push({
        open: slice[0],
        close: slice[slice.length - 1],
        high: Math.max(...slice),
        low: Math.min(...slice),
      });
    }
    return candles;
  }

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
    const rightMargin = 38; // space for price labels
    const w = totalW - rightMargin;
    const pad = 4; // top/bottom padding

    ctx.clearRect(0, 0, totalW, h);

    const candles = buildCandles(priceHistory);
    const allPrices = priceHistory;
    let min = Math.min(...allPrices) - 0.03;
    let max = Math.max(...allPrices) + 0.03;
    const range = max - min || 1;

    const toY = (p) => pad + (h - pad * 2) * (1 - (p - min) / range);

    // --- Horizontal level lines with right-side labels ---
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
      // Label on right
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.fillStyle = color;
      ctx.textAlign = "left";
      ctx.fillText(price.toFixed(2), w + 4, y + 3);
    }

    drawLevel(state.support_floor, "rgba(74,222,128,0.5)", [4, 4]);
    drawLevel(state.resistance_ceiling, "rgba(248,113,113,0.5)", [4, 4]);
    drawLevel(state.vwap, "rgba(96,165,250,0.5)", [2, 3]);

    // --- Price axis labels (high/low of visible range) ---
    ctx.font = "8px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#555";
    ctx.textAlign = "left";
    ctx.fillText(max.toFixed(2), w + 4, pad + 6);
    ctx.fillText(min.toFixed(2), w + 4, h - pad + 2);

    // --- Candles ---
    if (candles.length > 0) {
      const gap = 1;
      const candleW = Math.max(2, (w - gap * candles.length) / candles.length);

      for (let i = 0; i < candles.length; i++) {
        const c = candles[i];
        const x = i * (candleW + gap);
        const bullish = c.close >= c.open;
        const bodyColor = bullish ? "rgba(74,222,128,0.85)" : "rgba(248,113,113,0.85)";
        const wickColor = bullish ? "rgba(74,222,128,0.4)" : "rgba(248,113,113,0.4)";

        // Wick (high-low line)
        const wickX = x + candleW / 2;
        ctx.strokeStyle = wickColor;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(wickX, toY(c.high));
        ctx.lineTo(wickX, toY(c.low));
        ctx.stroke();

        // Body (open-close rect)
        const bodyTop = toY(Math.max(c.open, c.close));
        const bodyBot = toY(Math.min(c.open, c.close));
        const bodyH = Math.max(1, bodyBot - bodyTop);
        ctx.fillStyle = bodyColor;
        ctx.fillRect(x, bodyTop, candleW, bodyH);
      }
    }

    // --- Current price tag on right ---
    if (state.price > 0) {
      const curY = toY(state.price);
      // Small tag background
      ctx.fillStyle = "#f0f0f0";
      ctx.fillRect(w + 1, curY - 6, rightMargin - 2, 12);
      ctx.font = "bold 8px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#0a0a0a";
      ctx.textAlign = "left";
      ctx.fillText(state.price.toFixed(2), w + 4, curY + 3);
      // Dotted line across chart
      ctx.strokeStyle = "rgba(240,240,240,0.25)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(0, curY);
      ctx.lineTo(w, curY);
      ctx.stroke();
      ctx.setLineDash([]);
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

    // Also catch visibility change to immediately re-check timer
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && timerEndAt && !timerDone) {
        tickTimer();
      }
    });

    window.addEventListener("resize", drawChart);
  });
})();
