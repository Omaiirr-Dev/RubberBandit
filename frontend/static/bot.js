(function () {
  "use strict";

  // ---- State ----
  let activeTab = "live"; // "live" or "demo"
  let ws = null;
  let connected = false;
  let replayActive = false;

  const liveData = {
    dayPrices: [],
    markers: [],
    trades: [],
    state: null,
  };
  const demoData = {
    dayPrices: [],
    markers: [],
    trades: [],
    state: null,
  };

  // Chart hover
  let hoverX = -1;

  // ---- DOM refs ----
  const $ = (id) => document.getElementById(id);
  const dot = $("dot");
  const statusText = $("status-text");
  const tabLive = $("tab-live");
  const tabDemo = $("tab-demo");
  const statusPill = $("status-pill");
  const warmupTrack = $("warmup-track");
  const warmupFill = $("warmup-fill");
  const warmupPct = $("warmup-pct");
  const cashAmount = $("cash-amount");
  const cashPnl = $("cash-pnl");
  const positionCard = $("position-card");
  const posPnl = $("pos-pnl");
  const posShares = $("pos-shares");
  const posEntry = $("pos-entry");
  const posPnlPct = $("pos-pnl-pct");
  const statTrades = $("stat-trades");
  const statWinrate = $("stat-winrate");
  const statBest = $("stat-best");
  const statWorst = $("stat-worst");
  const floorEl = $("floor");
  const ceilEl = $("ceil");
  const modeEl = $("mode");
  const vwapEl = $("vwap");
  const signalBadge = $("signal-badge");
  const signalScore = $("signal-score");
  const trendBadge = $("trend-badge");
  const aiCard = $("ai-card");
  const aiBadge = $("ai-badge");
  const aiConf = $("ai-conf");
  const aiTimer = $("ai-timer");
  const aiReason = $("ai-reason");
  const gaugeFill = $("gauge-fill");
  const chartPriceLabel = $("chart-price-label");
  const tradeCount = $("trade-count");
  const tradeList = $("trade-list");
  const canvas = $("chart");
  const ctx = canvas.getContext("2d");

  // Replay DOM refs
  const replayBar = $("replay-bar");
  const replayLabel = $("replay-label");
  const replayFill = $("replay-fill");
  const replayInfo = $("replay-info");
  const replayBtn = $("replay-btn");

  // ---- Helpers ----
  function fmt(n) {
    if (n == null || n === 0) return "--";
    return n.toFixed(2);
  }
  function fmtMoney(n) {
    if (n == null) return "$0.00";
    const sign = n >= 0 ? "" : "-";
    return sign + "$" + Math.abs(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  function fmtPnl(pnl, pct) {
    const sign = pnl >= 0 ? "+" : "";
    return sign + fmtMoney(pnl) + " (" + (pnl >= 0 ? "+" : "") + pct.toFixed(2) + "%)";
  }
  function fmtTime(seconds) {
    if (seconds < 60) return Math.round(seconds) + "s";
    return Math.floor(seconds / 60) + "m" + Math.round(seconds % 60) + "s";
  }

  function activeData() {
    return activeTab === "live" ? liveData : demoData;
  }

  // ---- Tab switching ----
  window.switchTab = function (tab) {
    activeTab = tab;
    tabLive.className = "tab-btn" + (tab === "live" ? " active" : "");
    tabDemo.className = "tab-btn" + (tab === "demo" ? " active-demo" : "");
    render();
  };

  // ---- Replay controls ----
  window.startReplay = async function () {
    replayBtn.textContent = "LOADING...";
    replayBtn.disabled = true;
    try {
      const resp = await fetch("/api/replay", { method: "POST" });
      const result = await resp.json();
      if (!result.ok) {
        alert("Replay failed: " + (result.reason || "Unknown error"));
        replayBtn.textContent = "REPLAY TODAY";
        replayBtn.disabled = false;
      }
      // Button state will be updated by replay_start message
    } catch (e) {
      alert("Replay error: " + e.message);
      replayBtn.textContent = "REPLAY TODAY";
      replayBtn.disabled = false;
    }
  };

  window.stopReplay = async function () {
    try {
      await fetch("/api/replay/stop", { method: "POST" });
    } catch (e) {
      // ignore
    }
  };

  function setReplayUI(active) {
    replayActive = active;
    if (active) {
      replayBtn.textContent = "STOP";
      replayBtn.className = "replay-btn stop";
      replayBtn.disabled = false;
      replayBtn.onclick = window.stopReplay;
      replayBar.style.display = "flex";
    } else {
      replayBtn.textContent = "REPLAY TODAY";
      replayBtn.className = "replay-btn";
      replayBtn.disabled = false;
      replayBtn.onclick = window.startReplay;
      replayBar.style.display = "none";
    }
  }

  // ---- WebSocket ----
  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(proto + "//" + location.host + "/ws/bot");

    ws.onopen = function () {
      connected = true;
      dot.className = "dot connected";
      statusText.textContent = "CONNECTED";
    };

    ws.onclose = function () {
      connected = false;
      dot.className = "dot";
      statusText.textContent = "DISCONNECTED";
      setTimeout(connect, 3000);
    };

    ws.onerror = function () {
      ws.close();
    };

    ws.onmessage = function (evt) {
      const data = JSON.parse(evt.data);
      handleMessage(data);
    };
  }

  function handleMessage(data) {
    const type = data.type;

    if (type === "bot_update") {
      if (data.live) liveData.state = data.live;
      if (data.demo) demoData.state = data.demo;

      // Accumulate day prices
      if (data.live && data.live.price > 0 && !data.replay_ts) {
        liveData.dayPrices.push([Date.now(), data.live.price]);
        thinPrices(liveData);
      }
      if (data.demo && data.demo.price > 0) {
        // During replay: use market timestamp; otherwise: wall clock
        const ts = data.replay_ts || Date.now();
        demoData.dayPrices.push([ts, data.demo.price]);
        thinPrices(demoData);
      }

      // Update markers from recent trades
      if (data.live) updateMarkersFromState(liveData, data.live);
      if (data.demo) updateMarkersFromState(demoData, data.demo);

      // Update replay progress bar
      if (data.replay_progress != null) {
        replayFill.style.width = data.replay_progress + "%";
        const mkt = data.replay_market_time || 0;
        replayInfo.textContent = data.replay_progress.toFixed(0) + "% | " + mkt.toFixed(0) + "min";
      }

      render();

    } else if (type === "replay_loading") {
      // Show loading status
      replayBar.style.display = "flex";
      replayLabel.textContent = "LOADING";
      replayInfo.textContent = data.status || "Fetching...";
      replayFill.style.width = "0%";

    } else if (type === "replay_start") {
      // Clear demo data for fresh replay
      demoData.dayPrices = [];
      demoData.markers = [];
      demoData.trades = [];
      demoData.state = null;
      // Auto-switch to demo tab
      window.switchTab("demo");
      setReplayUI(true);
      replayLabel.textContent = "REPLAYING " + (data.source === "trades" ? "REAL TICKS" : "BARS");
      const est = data.est_replay_minutes || 0;
      replayInfo.textContent = "~" + Math.round(est) + "min | " + (data.total_ticks || 0).toLocaleString() + " ticks";

    } else if (type === "replay_end") {
      setReplayUI(false);
      // Show final summary in replay bar briefly
      replayBar.style.display = "flex";
      replayLabel.textContent = "REPLAY DONE";
      const pnl = data.final_pnl || 0;
      const pnlStr = (pnl >= 0 ? "+" : "") + "$" + Math.abs(pnl).toFixed(2);
      replayInfo.textContent = data.total_trades + " trades | " + (data.win_rate || 0) + "% win | " + pnlStr;
      replayFill.style.width = "100%";
      replayFill.style.background = pnl >= 0 ? "var(--green)" : "var(--red)";
      // Hide after 30 seconds
      setTimeout(function () {
        if (!replayActive) {
          replayBar.style.display = "none";
          replayFill.style.background = "var(--yellow)";
        }
      }, 30000);

    } else if (type === "replay_error") {
      setReplayUI(false);
      alert("Replay error: " + (data.reason || "Unknown"));

    } else if (type === "day_init_live") {
      liveData.dayPrices = data.prices || [];
    } else if (type === "day_init_demo") {
      demoData.dayPrices = data.prices || [];
    } else if (type === "trade_log_live") {
      liveData.trades = data.trades || [];
    } else if (type === "trade_log_demo") {
      demoData.trades = data.trades || [];
    } else if (type === "markers_live") {
      liveData.markers = data.markers || [];
    } else if (type === "markers_demo") {
      demoData.markers = data.markers || [];
    }
  }

  function thinPrices(store) {
    if (store.dayPrices.length > 3000) {
      const thinned = [];
      const step = 2;
      for (let i = 0; i < store.dayPrices.length; i += step) {
        thinned.push(store.dayPrices[i]);
      }
      thinned.push(store.dayPrices[store.dayPrices.length - 1]);
      store.dayPrices = thinned;
    }
  }

  function updateMarkersFromState(store, state) {
    if (!state.recent_trades) return;
    const trades = state.recent_trades;
    if (state.trade_count > store.trades.length) {
      for (const t of trades) {
        const exists = store.trades.find((x) => x.id === t.id);
        if (!exists) {
          store.trades.push(t);
        }
      }
    }
    // Rebuild markers from full trade list
    store.markers = [];
    for (const t of store.trades) {
      store.markers.push({ ts: Math.round(t.entry_time * 1000), price: t.entry_price, type: "BUY" });
      store.markers.push({ ts: Math.round(t.exit_time * 1000), price: t.exit_price, type: "SELL" });
    }
    if (state.in_position && state.position_entry_price > 0) {
      store.markers.push({ ts: Date.now() - 60000, price: state.position_entry_price, type: "BUY" });
    }
  }

  // ---- Render ----
  function render() {
    const d = activeData();
    const s = d.state;
    if (!s) return;

    // Status pill
    updateStatusPill(s.bot_status, s.warmup_pct);

    // Cash
    cashAmount.textContent = fmtMoney(s.total_value);
    const pnl = s.total_pnl || 0;
    const pnlPct = s.total_pnl_pct || 0;
    cashPnl.textContent = fmtPnl(pnl, pnlPct);
    cashPnl.className = "cash-pnl " + (pnl > 0.01 ? "profit" : pnl < -0.01 ? "loss" : "flat");

    // Position
    if (s.in_position) {
      positionCard.className = "position-card show" + (s.unrealized_pnl < 0 ? " losing" : "");
      const uPnl = s.unrealized_pnl || 0;
      const uPct = s.unrealized_pnl_pct || 0;
      posPnl.textContent = (uPnl >= 0 ? "+" : "") + fmtMoney(uPnl);
      posPnl.className = "pos-pnl " + (uPnl >= 0 ? "profit" : "loss");
      posShares.textContent = s.position_shares + " shares";
      posEntry.textContent = "$" + s.position_entry_price.toFixed(2);
      posPnlPct.textContent = "(" + (uPct >= 0 ? "+" : "") + uPct.toFixed(2) + "%)";
    } else {
      positionCard.className = "position-card";
    }

    // Stats
    statTrades.textContent = s.trade_count;
    const trades = d.trades;
    if (trades.length > 0) {
      const wins = trades.filter((t) => t.pnl > 0).length;
      statWinrate.textContent = Math.round((wins / trades.length) * 100) + "%";
      statWinrate.style.color = wins / trades.length >= 0.5 ? "var(--green)" : "var(--red)";
      const best = Math.max(...trades.map((t) => t.pnl));
      const worst = Math.min(...trades.map((t) => t.pnl));
      statBest.textContent = (best >= 0 ? "+" : "") + "$" + Math.abs(best).toFixed(0);
      statBest.style.color = best >= 0 ? "var(--green)" : "var(--red)";
      statWorst.textContent = (worst >= 0 ? "+" : "") + "$" + Math.abs(worst).toFixed(0);
      statWorst.style.color = worst >= 0 ? "var(--green)" : "var(--red)";
    } else {
      statWinrate.textContent = "--";
      statWinrate.style.color = "var(--text-dim)";
      statBest.textContent = "--";
      statBest.style.color = "var(--text-dim)";
      statWorst.textContent = "--";
      statWorst.style.color = "var(--text-dim)";
    }

    // Levels
    floorEl.textContent = fmt(s.support_floor);
    ceilEl.textContent = fmt(s.resistance_ceiling);
    modeEl.textContent = fmt(s.mode_price);
    vwapEl.textContent = fmt(s.vwap);

    // Signal
    const action = s.action || "WAIT";
    const score = s.signal_score || 0;
    signalBadge.textContent = action;
    signalBadge.className = "signal-badge " + action.toLowerCase();
    signalScore.textContent = score;
    signalScore.style.color = action === "BUY" ? "var(--green)" : action === "SELL" ? "var(--red)" : "var(--text-dim)";
    gaugeFill.style.width = score + "%";
    gaugeFill.className = "gauge-fill " + (score >= 70 ? "high" : score >= 40 ? "mid" : "low");

    // Trend
    const trend = s.trend || "down";
    trendBadge.textContent = trend === "up" ? "TREND UP" : "TREND DN";
    trendBadge.className = "trend-badge " + trend;

    // AI Brain
    renderAI(s);

    // Chart
    drawChart();

    // Trade log
    renderTradeLog();
  }

  function updateStatusPill(status, warmupPct) {
    const map = {
      WARMING_UP: ["WARMING UP", "warming"],
      FORMING_OR: ["FORMING OR", "warming"],
      WATCHING: ["WATCHING", "watching"],
      ENTERING: ["ENTERING", "entering"],
      IN_POSITION: ["IN POSITION", "in-position"],
      EXITING: ["EXITING", "exiting"],
      COOLING_DOWN: ["COOLING DOWN", "cooling"],
    };
    const [text, cls] = map[status] || ["--", "cooling"];
    statusPill.textContent = text;
    statusPill.className = "status-pill " + cls;

    // Warmup / OR formation progress
    if (status === "WARMING_UP" || status === "FORMING_OR") {
      warmupTrack.style.display = "";
      warmupPct = warmupPct || 0;
      warmupFill.style.width = warmupPct + "%";
      const label = status === "FORMING_OR" ? warmupPct + "% OR" : warmupPct + "%";
      document.getElementById("warmup-pct").textContent = label;
      document.getElementById("warmup-pct").style.display = "";
    } else {
      warmupTrack.style.display = "none";
      document.getElementById("warmup-pct").style.display = "none";
    }
  }

  // ---- Bot Brain (pattern detection) ----
  function renderAI(s) {
    if (!aiCard) return;
    aiCard.style.display = "";
    const rec = s.ai_recommendation || "WAIT";
    const reason = s.ai_reason || "Warming up...";
    const confidence = s.ai_confidence || 0;

    aiCard.className = "ai-card active";

    // Map pattern status to badge
    if (rec === "FORMING") {
      aiBadge.textContent = "FORMING";
      aiBadge.className = "ai-badge scanning";
    } else if (rec === "SCAN") {
      aiBadge.textContent = "SCAN";
      aiBadge.className = "ai-badge scanning";
    } else if (rec === "BUY") {
      aiBadge.textContent = "BUY";
      aiBadge.className = "ai-badge buy";
    } else {
      aiBadge.textContent = "WAIT";
      aiBadge.className = "ai-badge hold";
    }

    aiConf.textContent = Math.round(confidence * 100) + "%";
    // Show strategy label and ORB info
    let extra = "";
    if (s.strategy === "orb") {
      extra = " [ORB]";
      if (s.or_high && s.or_low) {
        extra += " H:" + s.or_high.toFixed(2) + " L:" + s.or_low.toFixed(2);
      }
      if (s.trailing_stop) extra += " TS:" + s.trailing_stop.toFixed(2);
      if (s.atr_ready) extra += " ATR:" + s.atr.toFixed(2);
    }
    aiTimer.textContent = extra;
    aiReason.textContent = reason;
  }

  // ---- Chart ----
  function drawChart() {
    const d = activeData();
    const prices = d.dayPrices;
    const markers = d.markers;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    if (prices.length < 2) {
      ctx.fillStyle = "#888";
      ctx.font = "11px 'JetBrains Mono'";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for data...", W / 2, H / 2);
      return;
    }

    const pad = { top: 10, right: 50, bottom: 20, left: 5 };
    const cw = W - pad.left - pad.right;
    const ch = H - pad.top - pad.bottom;

    // Price range
    let minP = Infinity, maxP = -Infinity;
    for (const p of prices) {
      const price = p[1];
      if (price < minP) minP = price;
      if (price > maxP) maxP = price;
    }
    // Include markers in range
    for (const m of markers) {
      if (m.price < minP) minP = m.price;
      if (m.price > maxP) maxP = m.price;
    }
    // Include ORB levels in chart range
    const s_range = d.state;
    if (s_range && s_range.or_high && s_range.or_high > maxP) maxP = s_range.or_high;
    if (s_range && s_range.or_low && s_range.or_low < minP) minP = s_range.or_low;
    const pRange = maxP - minP || 1;
    const margin = pRange * 0.08;
    minP -= margin;
    maxP += margin;

    // Time range
    const tMin = prices[0][0];
    const tMax = prices[prices.length - 1][0];
    const tRange = tMax - tMin || 1;

    function x(ts) { return pad.left + ((ts - tMin) / tRange) * cw; }
    function y(price) { return pad.top + (1 - (price - minP) / (maxP - minP)) * ch; }

    // Background
    ctx.clearRect(0, 0, W, H);

    // Grid lines (horizontal)
    const steps = 4;
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= steps; i++) {
      const py = pad.top + (ch / steps) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, py);
      ctx.lineTo(W - pad.right, py);
      ctx.stroke();
      // Price label
      const pLabel = maxP - ((maxP - minP) / steps) * i;
      ctx.fillStyle = "#555";
      ctx.font = "9px 'JetBrains Mono'";
      ctx.textAlign = "left";
      ctx.fillText(pLabel.toFixed(2), W - pad.right + 4, py + 3);
    }

    // Time labels
    ctx.fillStyle = "#555";
    ctx.font = "8px 'JetBrains Mono'";
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) {
      const ts = tMin + (tRange / 4) * i;
      const date = new Date(ts);
      const hh = String(date.getHours()).padStart(2, "0");
      const mm = String(date.getMinutes()).padStart(2, "0");
      ctx.fillText(hh + ":" + mm, x(ts), H - 4);
    }

    // Price line
    ctx.beginPath();
    ctx.moveTo(x(prices[0][0]), y(prices[0][1]));
    for (let i = 1; i < prices.length; i++) {
      const prevP = prices[i - 1][1];
      const currP = prices[i][1];
      ctx.strokeStyle = currP >= prevP ? "#4ade80" : "#f87171";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(x(prices[i - 1][0]), y(prevP));
      ctx.lineTo(x(prices[i][0]), y(currP));
      ctx.stroke();
    }

    // ORB overlay lines (OR High, OR Low, trailing stop)
    const s_chart = d.state;
    if (s_chart && s_chart.strategy === "orb") {
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      // OR High (green dashed)
      if (s_chart.or_high) {
        ctx.strokeStyle = "rgba(74, 222, 128, 0.5)";
        ctx.beginPath();
        ctx.moveTo(pad.left, y(s_chart.or_high));
        ctx.lineTo(W - pad.right, y(s_chart.or_high));
        ctx.stroke();
      }
      // OR Low (red dashed)
      if (s_chart.or_low) {
        ctx.strokeStyle = "rgba(248, 113, 113, 0.5)";
        ctx.beginPath();
        ctx.moveTo(pad.left, y(s_chart.or_low));
        ctx.lineTo(W - pad.right, y(s_chart.or_low));
        ctx.stroke();
      }
      // Trailing stop (orange dashed)
      if (s_chart.trailing_stop && s_chart.in_position) {
        ctx.strokeStyle = "rgba(251, 191, 36, 0.7)";
        ctx.beginPath();
        ctx.moveTo(pad.left, y(s_chart.trailing_stop));
        ctx.lineTo(W - pad.right, y(s_chart.trailing_stop));
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    // Trade markers
    for (const m of markers) {
      if (m.ts < tMin || m.ts > tMax + tRange * 0.05) continue;
      const mx = x(m.ts);
      const my = y(m.price);
      ctx.beginPath();
      if (m.type === "BUY") {
        // Green triangle up
        ctx.fillStyle = "#4ade80";
        ctx.moveTo(mx, my - 6);
        ctx.lineTo(mx - 4, my + 2);
        ctx.lineTo(mx + 4, my + 2);
      } else {
        // Red triangle down
        ctx.fillStyle = "#f87171";
        ctx.moveTo(mx, my + 6);
        ctx.lineTo(mx - 4, my - 2);
        ctx.lineTo(mx + 4, my - 2);
      }
      ctx.closePath();
      ctx.fill();
    }

    // Hover crosshair
    if (hoverX >= 0 && hoverX >= pad.left && hoverX <= W - pad.right) {
      const ratio = (hoverX - pad.left) / cw;
      const idx = Math.round(ratio * (prices.length - 1));
      const clamped = Math.max(0, Math.min(prices.length - 1, idx));
      const pt = prices[clamped];
      const px = x(pt[0]);
      const py = y(pt[1]);

      // Vertical line
      ctx.strokeStyle = "rgba(255,255,255,0.2)";
      ctx.lineWidth = 0.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(px, pad.top);
      ctx.lineTo(px, pad.top + ch);
      ctx.stroke();

      // Horizontal line
      ctx.beginPath();
      ctx.moveTo(pad.left, py);
      ctx.lineTo(W - pad.right, py);
      ctx.stroke();
      ctx.setLineDash([]);

      // Dot
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.fill();

      // Tooltip
      const date = new Date(pt[0]);
      const hh = String(date.getHours()).padStart(2, "0");
      const mm = String(date.getMinutes()).padStart(2, "0");
      const ss = String(date.getSeconds()).padStart(2, "0");
      const label = "$" + pt[1].toFixed(2) + "  " + hh + ":" + mm + ":" + ss;
      chartPriceLabel.textContent = label;
    } else {
      const d2 = activeData();
      const s = d2.state;
      if (s && s.price > 0) {
        chartPriceLabel.textContent = "$" + s.price.toFixed(2);
      } else {
        chartPriceLabel.textContent = "";
      }
    }
  }

  // ---- Trade log ----
  function renderTradeLog() {
    const d = activeData();
    const trades = d.trades;
    tradeCount.textContent = trades.length + " trade" + (trades.length !== 1 ? "s" : "");

    if (trades.length === 0) {
      tradeList.innerHTML = '<div class="no-trades">No trades yet</div>';
      return;
    }

    // Show newest first
    let html = "";
    for (let i = trades.length - 1; i >= 0; i--) {
      const t = trades[i];
      const pnlClass = t.pnl >= 0 ? "profit" : "loss";
      const reasonMap = {
        TAKE_PROFIT: ["TP", "tp"],
        STOP_LOSS: ["SL", "sl"],
        TIME_LIMIT: ["TL", "tl"],
        SELL_SIGNAL: ["SS", "ss"],
        EMA_CROSS: ["EC", "tp"],
        INDICATOR_EXIT: ["IE", "tp"],
        AI_PROFIT: ["AI$", "tp"],
        AI_SELL: ["AI", "ss"],
        PARTIAL_TP: ["PT", "tp"],
        TRAILING_STOP: ["TS", "sl"],
        EOD_EXIT: ["EOD", "tl"],
        ORB_STOP: ["OS", "sl"],
      };
      const [reasonText, reasonCls] = reasonMap[t.exit_reason] || [t.exit_reason || "?", "tl"];
      const holdStr = fmtTime(t.hold_seconds);
      const pnlSign = t.pnl >= 0 ? "+" : "";

      html += '<div class="trade-item">' +
        '<span class="trade-id">#' + t.id + '</span>' +
        '<span class="trade-prices">$' + t.entry_price.toFixed(2) + ' → $' + t.exit_price.toFixed(2) + '</span>' +
        '<span class="trade-shares">' + t.shares + 'sh</span>' +
        '<span class="trade-hold">' + holdStr + '</span>' +
        '<span class="trade-reason ' + reasonCls + '">' + reasonText + '</span>' +
        '<span class="trade-pnl ' + pnlClass + '">' + pnlSign + '$' + Math.abs(t.pnl).toFixed(2) + '</span>' +
        '</div>';
    }
    tradeList.innerHTML = html;
  }

  // ---- Chart interaction ----
  canvas.addEventListener("mousemove", function (e) {
    const rect = canvas.getBoundingClientRect();
    hoverX = e.clientX - rect.left;
    drawChart();
  });
  canvas.addEventListener("mouseleave", function () {
    hoverX = -1;
    drawChart();
  });
  canvas.addEventListener("touchmove", function (e) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    hoverX = e.touches[0].clientX - rect.left;
    drawChart();
  }, { passive: false });
  canvas.addEventListener("touchend", function () {
    hoverX = -1;
    drawChart();
  });

  // ResizeObserver
  new ResizeObserver(function () {
    drawChart();
  }).observe(canvas.parentElement);

  // ---- Init ----
  connect();
})();
