// RubberBand Frontend
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  let ws;
  let state = {};
  let priceHistory = [];
  const MAX_CHART_POINTS = 120;

  // Day chart data: [{ts, price}, ...]
  let dayPrices = [];
  const MAX_DAY_POINTS = 2000;
  let chartMode = "live"; // "live" or "day"

  // Hover crosshair state for day chart
  let hoverX = null; // CSS pixel X relative to canvas, or null

  // Zoom & pan state for day chart (frozen absolute bounds)
  let zoomLevel = 1.0;
  let viewMinT = null;   // frozen left bound (ms timestamp), null = show all
  let viewMaxT = null;   // frozen right bound
  let isDragging = false;
  let dragStartX = 0;
  let dragStartMinT = 0;
  let dragStartMaxT = 0;

  // Click markers: [{ts, price, type: "A"|"B"}]
  let chartMarkers = [];
  let nextMarkerType = "A";

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
      const data = JSON.parse(e.data);

      // Handle day_init (one-time backfill on connect)
      if (data.type === "day_init") {
        dayPrices = data.prices.map(([ts, price]) => ({ ts, price }));
        if (chartMode === "day") drawDayChart();
        return;
      }

      state = data;
      if (state.price > 0) {
        priceHistory.push(state.price);
        if (priceHistory.length > MAX_CHART_POINTS) priceHistory.shift();
        // Also accumulate to day chart
        dayPrices.push({ ts: Date.now(), price: state.price });
        if (dayPrices.length > MAX_DAY_POINTS) {
          // Thin older half: keep every other point from first half
          const half = Math.floor(dayPrices.length / 2);
          const thinned = [];
          for (let i = 0; i < half; i += 2) thinned.push(dayPrices[i]);
          dayPrices = thinned.concat(dayPrices.slice(half));
        }
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
        dayPrices = [];
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

  // ---- Price Alerts ----

  function checkAlerts(price) {
    const el = $("#price");
    if (!el || price <= 0) return;
    const highVal = parseFloat($("#alert-high").value);
    const lowVal = parseFloat($("#alert-low").value);
    el.classList.remove("flash-green", "flash-red");
    if (highVal && Math.abs(price - highVal) <= 0.10) {
      el.classList.add("flash-green");
    } else if (lowVal && Math.abs(price - lowVal) <= 0.10) {
      el.classList.add("flash-red");
    }
  }

  // ---- Top 5 + Day High ----

  function renderTop5(top5) {
    const grid = $("#top5-grid");
    if (!grid || !top5) return;
    // Color gradient: index 0 = highest price = green, last = red
    const colors = ["var(--green)", "#86efac", "var(--yellow)", "#fca5a5", "var(--red)"];
    let html = "";
    for (let i = 0; i < 5; i++) {
      if (i < top5.length) {
        const item = top5[i];
        html += `<div class="top5-item"><span class="top5-price" style="color:${colors[i]}">$${item.price.toFixed(2)}</span><span class="top5-count">${item.count}</span></div>`;
      } else {
        html += `<div class="top5-item"><span class="top5-price" style="color:var(--text-dim)">---</span><span class="top5-count">0</span></div>`;
      }
    }
    grid.innerHTML = html;
  }

  function renderDayHighLow(dayHigh, dayLow) {
    const hEl = $("#day-high");
    const lEl = $("#day-low");
    if (hEl) hEl.textContent = dayHigh > 0 ? `$${dayHigh.toFixed(2)}` : "---";
    if (lEl) lEl.textContent = dayLow > 0 ? `$${dayLow.toFixed(2)}` : "---";
  }

  // ---- Render ----

  function render(s) {
    // Demo indicator
    if (s.demo !== undefined) updateDemoUI(s.demo);

    // Context indicator
    if (s.context_active !== undefined) updateContextUI(s);

    // Price
    $("#price").textContent = s.price > 0 ? `$${s.price.toFixed(2)}` : "---";

    // Price alerts
    checkAlerts(s.price);

    // Top 5 + Day High/Low
    if (s.top5) renderTop5(s.top5);
    renderDayHighLow(s.day_high || 0, s.day_low || 0);

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
    if (chartMode === "live") drawChart();
    else drawDayChart();
  }

  // ---- Chart (Live) ----

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

  // ---- Chart (Day / Since Open) ----

  function getVisibleDayWindow() {
    if (dayPrices.length < 2) return { data: dayPrices, minT: 0, maxT: 1 };
    const allMinT = dayPrices[0].ts;
    const allMaxT = dayPrices[dayPrices.length - 1].ts;
    // When frozen bounds are set, use them directly (view doesn't shift with new data)
    if (viewMinT !== null && viewMaxT !== null) {
      const visible = dayPrices.filter(d => d.ts >= viewMinT && d.ts <= viewMaxT);
      if (visible.length < 2) return { data: dayPrices, minT: allMinT, maxT: allMaxT };
      return { data: visible, minT: viewMinT, maxT: viewMaxT };
    }
    // No zoom — show everything
    return { data: dayPrices, minT: allMinT, maxT: allMaxT };
  }

  function drawDayChart() {
    const canvas = $("#chart");
    if (!canvas || dayPrices.length < 2) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    if (rect.width < 10 || rect.height < 10) {
      requestAnimationFrame(drawDayChart);
      return;
    }

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const totalW = rect.width;
    const h = rect.height;
    const rm = 38;
    const bm = 14; // bottom margin for time labels
    const w = totalW - rm;
    const ch = h - bm; // chart height

    ctx.clearRect(0, 0, totalW, h);

    // Visible window (zoom/pan)
    const { data: visData, minT, maxT } = getVisibleDayWindow();
    if (visData.length < 2) return;

    const visPrices = visData.map(d => d.price);
    const minP = Math.min(...visPrices) - 0.05;
    const maxP = Math.max(...visPrices) + 0.05;
    const rangeP = maxP - minP || 1;
    const toY = (p) => ch - ((p - minP) / rangeP) * ch;

    const rangeT = maxT - minT || 1;
    const toX = (t) => ((t - minT) / rangeT) * w;

    // Day high/low lines
    const curY = state.price > 0 ? toY(state.price) : -100;
    const usedLabelYs = [curY];

    function drawLevel(price, color, dash) {
      if (price <= 0) return;
      const y = toY(price);
      if (y < 0 || y > ch) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
      ctx.setLineDash([]);
      const tooClose = usedLabelYs.some(ly => Math.abs(ly - y) < 12);
      if (!tooClose) {
        ctx.font = "8px 'JetBrains Mono', monospace";
        ctx.fillStyle = color;
        ctx.textAlign = "left";
        ctx.fillText(price.toFixed(2), w + 3, y + 3);
        usedLabelYs.push(y);
      }
    }

    // Draw context lines on day chart too
    if (state.context_active) {
      drawLevel(state.context_high, "rgba(34,211,238,0.6)", [8, 4]);
      drawLevel(state.context_low, "rgba(34,211,238,0.6)", [8, 4]);
    }

    // High/low labels
    if (!usedLabelYs.some(ly => Math.abs(ly - 6) < 12)) {
      ctx.font = "7px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#444";
      ctx.textAlign = "left";
      ctx.fillText(maxP.toFixed(2), w + 3, 8);
    }
    if (!usedLabelYs.some(ly => Math.abs(ly - (ch - 4)) < 12)) {
      ctx.font = "7px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#444";
      ctx.textAlign = "left";
      ctx.fillText(minP.toFixed(2), w + 3, ch - 2);
    }

    // Time axis labels (3-4 labels spread across)
    ctx.font = "7px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#444";
    ctx.textAlign = "center";
    const labelCount = 4;
    for (let i = 0; i < labelCount; i++) {
      const t = minT + (rangeT * i) / (labelCount - 1);
      const x = toX(t);
      const d = new Date(t);
      const hh = d.getHours().toString().padStart(2, "0");
      const mm = d.getMinutes().toString().padStart(2, "0");
      ctx.fillText(`${hh}:${mm}`, x, h - 2);
    }

    // Price line — green/red segments
    ctx.lineWidth = 1.5;
    for (let i = 1; i < visData.length; i++) {
      const x0 = toX(visData[i - 1].ts);
      const y0 = toY(visData[i - 1].price);
      const x1 = toX(visData[i].ts);
      const y1 = toY(visData[i].price);
      ctx.beginPath();
      ctx.strokeStyle = visData[i].price >= visData[i - 1].price
        ? "rgba(74,222,128,0.9)"
        : "rgba(248,113,113,0.9)";
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }

    // Gradient fill
    ctx.beginPath();
    for (let i = 0; i < visData.length; i++) {
      const xp = toX(visData[i].ts);
      const yp = toY(visData[i].price);
      if (i === 0) ctx.moveTo(xp, yp);
      else ctx.lineTo(xp, yp);
    }
    ctx.lineTo(toX(visData[visData.length - 1].ts), ch);
    ctx.lineTo(toX(visData[0].ts), ch);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, ch);
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

    // ---- Click markers ----
    for (const marker of chartMarkers) {
      const mx = toX(marker.ts);
      const my = toY(marker.price);
      if (mx >= 0 && mx <= w && my >= 0 && my <= ch) {
        ctx.beginPath();
        ctx.arc(mx, my, 5, 0, Math.PI * 2);
        ctx.fillStyle = marker.type === "A" ? "#f87171" : "#4ade80";
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // Connecting line + stats between two markers
    if (chartMarkers.length === 2) {
      const mA = chartMarkers[0];
      const mB = chartMarkers[1];
      const ax = toX(mA.ts), ay = toY(mA.price);
      const bx = toX(mB.ts), by = toY(mB.price);

      ctx.strokeStyle = "rgba(255,255,255,0.5)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
      ctx.setLineDash([]);

      const dollarChg = mB.price - mA.price;
      const pctChg = (dollarChg / mA.price) * 100;
      const timeDiffSec = Math.abs(mB.ts - mA.ts) / 1000;
      let timeStr;
      if (timeDiffSec < 60) timeStr = Math.round(timeDiffSec) + "s";
      else if (timeDiffSec < 3600) timeStr = Math.floor(timeDiffSec / 60) + "m " + Math.round(timeDiffSec % 60) + "s";
      else timeStr = Math.floor(timeDiffSec / 3600) + "h " + Math.floor((timeDiffSec % 3600) / 60) + "m";

      const sign = dollarChg >= 0 ? "+" : "";
      const statsText = `${sign}$${dollarChg.toFixed(2)}  ${sign}${pctChg.toFixed(2)}%  ${timeStr}`;

      const midX = (ax + bx) / 2;
      const midY = Math.min(ay, by) - 14;

      ctx.font = "bold 9px 'JetBrains Mono', monospace";
      const stw = ctx.measureText(statsText).width + 10;
      const sth = 16;
      let stx = midX - stw / 2;
      if (stx < 0) stx = 0;
      if (stx + stw > w) stx = w - stw;
      let sty = midY;
      if (sty < 0) sty = Math.max(ay, by) + 8;

      ctx.fillStyle = "rgba(17,17,17,0.92)";
      ctx.beginPath();
      ctx.roundRect(stx, sty, stw, sth, 3);
      ctx.fill();
      ctx.strokeStyle = dollarChg >= 0 ? "rgba(74,222,128,0.6)" : "rgba(248,113,113,0.6)";
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = dollarChg >= 0 ? "#4ade80" : "#f87171";
      ctx.textAlign = "left";
      ctx.fillText(statsText, stx + 5, sty + 11);
    }

    // Hover crosshair
    if (hoverX !== null && hoverX >= 0 && hoverX <= w) {
      const hoverT = minT + (hoverX / w) * rangeT;
      let nearest = 0;
      let bestDist = Infinity;
      for (let i = 0; i < visData.length; i++) {
        const dist = Math.abs(visData[i].ts - hoverT);
        if (dist < bestDist) { bestDist = dist; nearest = i; }
      }
      const pt = visData[nearest];
      const px = toX(pt.ts);
      const py = toY(pt.price);

      ctx.strokeStyle = "rgba(240,240,240,0.3)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, ch);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, py);
      ctx.lineTo(w, py);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fillStyle = "#f0f0f0";
      ctx.fill();

      const d = new Date(pt.ts);
      const hh = d.getHours().toString().padStart(2, "0");
      const mm = d.getMinutes().toString().padStart(2, "0");
      const ss = d.getSeconds().toString().padStart(2, "0");
      const label = `$${pt.price.toFixed(2)}  ${hh}:${mm}:${ss}`;
      ctx.font = "bold 9px 'JetBrains Mono', monospace";
      const tw = ctx.measureText(label).width + 8;
      const th = 16;
      let tx = px - tw / 2;
      if (tx < 0) tx = 0;
      if (tx + tw > w) tx = w - tw;
      let ty = py - th - 6;
      if (ty < 0) ty = py + 8;

      ctx.fillStyle = "rgba(17,17,17,0.9)";
      ctx.beginPath();
      ctx.roundRect(tx, ty, tw, th, 3);
      ctx.fill();
      ctx.strokeStyle = "rgba(240,240,240,0.3)";
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = "#f0f0f0";
      ctx.textAlign = "left";
      ctx.fillText(label, tx + 4, ty + 11);
    }
  }

  // ---- Chart Tab Switching ----

  function switchChart(mode) {
    chartMode = mode;
    $("#tab-live").classList.toggle("active", mode === "live");
    $("#tab-day").classList.toggle("active", mode === "day");
    const wipeBtn = document.getElementById("wipe-markers-btn");
    if (wipeBtn) wipeBtn.style.display = mode === "day" ? "" : "none";
    if (mode === "live") {
      zoomLevel = 1.0;
      viewMinT = null;
      viewMaxT = null;
      drawChart();
    } else {
      drawDayChart();
    }
  }

  // ---- Timer ----
  // Uses Date.now() timestamps so the countdown stays accurate even if
  // the phone goes to the homescreen and iOS throttles setInterval.

  let timerUnit = "M"; // M = minutes, S = seconds

  function startTimer() {
    const val = parseInt($("#input-mins").value);
    if (!val || val < 1) return;
    const ms = timerUnit === "S" ? val * 1000 : val * 60 * 1000;

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
    timerEndAt = Date.now() + ms;
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
    $("#price").classList.remove("flash-green", "flash-red");
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

    // Timer unit toggle (M <-> S)
    $("#timer-unit").addEventListener("click", () => {
      timerUnit = timerUnit === "M" ? "S" : "M";
      $("#timer-unit").textContent = timerUnit;
    });

    $("#demo-btn").addEventListener("click", () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ cmd: "toggle_demo" }));
      }
    });

    // Context: unit toggle (M <-> S)
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

    // Chart tab switching
    $("#tab-live").addEventListener("click", () => switchChart("live"));
    $("#tab-day").addEventListener("click", () => switchChart("day"));

    // Day chart hover crosshair (mouse + touch)
    const chartCanvas = $("#chart");
    function handleHover(clientX) {
      if (chartMode !== "day" || dayPrices.length < 2) return;
      const rect = chartCanvas.getBoundingClientRect();
      hoverX = clientX - rect.left;
      drawDayChart();
    }
    chartCanvas.addEventListener("mousemove", (e) => {
      if (!isDragging) handleHover(e.clientX);
    });
    chartCanvas.addEventListener("mouseleave", () => {
      hoverX = null;
      if (chartMode === "day") drawDayChart();
    });
    chartCanvas.addEventListener("touchmove", (e) => {
      e.preventDefault();
      handleHover(e.touches[0].clientX);
    }, { passive: false });
    chartCanvas.addEventListener("touchend", () => {
      hoverX = null;
      if (chartMode === "day") drawDayChart();
    });

    // Scroll-to-zoom on day chart — freezes view bounds
    chartCanvas.addEventListener("wheel", (e) => {
      if (chartMode !== "day" || dayPrices.length < 2) return;
      e.preventDefault();
      const rect = chartCanvas.getBoundingClientRect();
      const rm = 38;
      const cw = rect.width - rm;
      const mouseX = e.clientX - rect.left;
      const allMinT = dayPrices[0].ts;
      const allMaxT = dayPrices[dayPrices.length - 1].ts;
      const totalRange = allMaxT - allMinT;
      if (totalRange <= 0) return;
      // Current visible bounds
      const curMinT = viewMinT !== null ? viewMinT : allMinT;
      const curMaxT = viewMaxT !== null ? viewMaxT : allMaxT;
      const curRange = curMaxT - curMinT;
      // Timestamp under mouse cursor
      const mouseT = curMinT + (mouseX / cw) * curRange;
      // Fraction of view left of cursor (anchor point)
      const frac = (mouseT - curMinT) / curRange;
      // Apply zoom
      const zoomFactor = e.deltaY < 0 ? 1.3 : 1 / 1.3;
      const newZoom = Math.max(1.0, Math.min(50.0, zoomLevel * zoomFactor));
      if (newZoom === zoomLevel) return;
      zoomLevel = newZoom;
      if (zoomLevel <= 1.01) {
        // Fully zoomed out — unfreeze
        zoomLevel = 1.0;
        viewMinT = null;
        viewMaxT = null;
      } else {
        // New visible range, anchored at mouse position
        const newRange = totalRange / zoomLevel;
        viewMinT = mouseT - frac * newRange;
        viewMaxT = mouseT + (1 - frac) * newRange;
        // Clamp to data bounds
        if (viewMinT < allMinT) { viewMinT = allMinT; viewMaxT = allMinT + newRange; }
        if (viewMaxT > allMaxT) { viewMaxT = allMaxT; viewMinT = allMaxT - newRange; }
        if (viewMinT < allMinT) viewMinT = allMinT;
      }
      drawDayChart();
    }, { passive: false });

    // Click-to-mark on day chart
    let mouseDownPos = null;
    chartCanvas.addEventListener("mousedown", (e) => {
      mouseDownPos = { x: e.clientX, y: e.clientY };
      if (chartMode === "day" && viewMinT !== null) {
        isDragging = true;
        dragStartX = e.clientX;
        dragStartMinT = viewMinT;
        dragStartMaxT = viewMaxT;
        chartCanvas.style.cursor = "grabbing";
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      const rect = chartCanvas.getBoundingClientRect();
      const rm = 38;
      const cw = rect.width - rm;
      const allMinT = dayPrices[0].ts;
      const allMaxT = dayPrices[dayPrices.length - 1].ts;
      const visibleRange = dragStartMaxT - dragStartMinT;
      const dx = e.clientX - dragStartX;
      const dtPerPx = visibleRange / cw;
      let newMin = dragStartMinT - dx * dtPerPx;
      let newMax = dragStartMaxT - dx * dtPerPx;
      // Clamp to data bounds
      if (newMin < allMinT) { newMin = allMinT; newMax = allMinT + visibleRange; }
      if (newMax > allMaxT) { newMax = allMaxT; newMin = allMaxT - visibleRange; }
      if (newMin < allMinT) newMin = allMinT;
      viewMinT = newMin;
      viewMaxT = newMax;
      drawDayChart();
    });

    window.addEventListener("mouseup", (e) => {
      const wasDragging = isDragging;
      if (isDragging) {
        isDragging = false;
        chartCanvas.style.cursor = "";
      }
      // Click detection: only place marker if mouse didn't move much
      if (mouseDownPos && chartMode === "day" && dayPrices.length >= 2) {
        const dx = Math.abs(e.clientX - mouseDownPos.x);
        const dy = Math.abs(e.clientY - mouseDownPos.y);
        if (dx < 5 && dy < 5) {
          // This is a click, not a drag
          const rect = chartCanvas.getBoundingClientRect();
          const rm = 38;
          const cw = rect.width - rm;
          const clickX = mouseDownPos.x - rect.left;
          if (clickX >= 0 && clickX <= cw) {
            const { data: visD, minT: vMinT, maxT: vMaxT } = getVisibleDayWindow();
            const vRangeT = vMaxT - vMinT || 1;
            const clickT = vMinT + (clickX / cw) * vRangeT;
            let nearest = 0;
            let bestDist = Infinity;
            for (let i = 0; i < visD.length; i++) {
              const dist = Math.abs(visD[i].ts - clickT);
              if (dist < bestDist) { bestDist = dist; nearest = i; }
            }
            const pt = visD[nearest];
            if (nextMarkerType === "A") {
              chartMarkers = [{ ts: pt.ts, price: pt.price, type: "A" }];
              nextMarkerType = "B";
            } else {
              chartMarkers.push({ ts: pt.ts, price: pt.price, type: "B" });
              nextMarkerType = "A";
            }
            drawDayChart();
          }
        }
      }
      mouseDownPos = null;
    });

    // Wipe markers button
    const wipeBtn = document.getElementById("wipe-markers-btn");
    if (wipeBtn) {
      wipeBtn.addEventListener("click", () => {
        chartMarkers = [];
        nextMarkerType = "A";
        zoomLevel = 1.0;
        viewMinT = null;
        viewMaxT = null;
        if (chartMode === "day") drawDayChart();
      });
    }

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
      new ResizeObserver(() => {
        if (chartMode === "live") drawChart();
        else drawDayChart();
      }).observe(chartWrap);
    } else {
      window.addEventListener("resize", () => {
        if (chartMode === "live") drawChart();
        else drawDayChart();
      });
    }
  });
})();
