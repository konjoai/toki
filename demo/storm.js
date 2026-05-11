/* ============================================================
   Toki — atmospheric system (storm + lightning + rain + wind +
   model-shape spring physics + aura rings).

   Usage:
     <div class="storm-host" id="hero">
       <div class="storm-content"> ...page content... </div>
     </div>
     initStorm(document.getElementById("hero"));

   API:
     initStorm(rootEl, { rainCount=110, windCount=10, ambientLightning=true })
     setStormIntensity(rootEl, "off"|"low"|"normal"|"high")
     triggerLightning(rootEl)
     mountSpringShape(wrapEl)         → returns Spring controller
     ShapeCtl.setVerdict("blocked"|"bypassed"|"warn"|"idle")
     ShapeCtl.setStreak(n)            → re-renders aura tiers
     ShapeCtl.bump()                  → manually kick the spring

   Pure ES5/6, no module system, no deps. Idempotent: safe to
   call initStorm() twice (it tears the previous atmosphere down
   first).
   ============================================================ */
(function (global) {
  "use strict";

  // ---------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------
  const rand = (a, b) => a + Math.random() * (b - a);
  const ri   = (a, b) => Math.floor(rand(a, b + 1));
  const NS   = "http://www.w3.org/2000/svg";

  function clearChildrenWithClass(parent, cls) {
    [...parent.querySelectorAll("." + cls)].forEach(n => n.remove());
  }

  // ---------------------------------------------------------
  // Init
  // ---------------------------------------------------------
  function initStorm(rootEl, opts) {
    if (!rootEl) return;
    opts = opts || {};
    rootEl.classList.add("storm-host");
    if (!rootEl.dataset.intensity) rootEl.dataset.intensity = "normal";

    // Wipe any prior atmosphere — initStorm is idempotent.
    ["storm-clouds", "storm-rain", "storm-wind", "storm-lightning", "storm-flash"]
      .forEach(c => [...rootEl.children].forEach(ch => { if (ch.classList && ch.classList.contains(c)) ch.remove(); }));

    // Cloud layers — 7 of them, drifting at varied speeds via CSS keyframes.
    const clouds = document.createElement("div");
    clouds.className = "storm-clouds";
    clouds.innerHTML = '<div class="cloud l1"></div><div class="cloud l2"></div><div class="cloud l3"></div><div class="cloud l4"></div><div class="cloud l5"></div><div class="cloud l6"></div><div class="cloud l7"></div>';
    rootEl.appendChild(clouds);

    // Rainfall.
    const rainCount = opts.rainCount ?? 110;
    const rain = document.createElement("div");
    rain.className = "storm-rain";
    for (let i = 0; i < rainCount; i++) {
      const drop = document.createElement("div");
      drop.className = "rain-drop";
      drop.style.left = rand(-5, 105) + "%";
      drop.style.height = ri(60, 120) + "px";
      drop.style.width = (Math.random() < .15 ? 2 : 1) + "px";
      drop.style.opacity = rand(.45, .9);
      const dur = rand(.45, .9).toFixed(2);
      const delay = rand(0, 1.2).toFixed(2);
      drop.style.animation = `fall ${dur}s linear ${delay}s infinite`;
      rain.appendChild(drop);
    }
    rootEl.appendChild(rain);

    // Wind streaks — sparse horizontal sweeps.
    const windCount = opts.windCount ?? 10;
    const wind = document.createElement("div");
    wind.className = "storm-wind";
    for (let i = 0; i < windCount; i++) {
      const w = document.createElement("div");
      w.className = "wind-streak";
      w.style.top = rand(0, 100) + "%";
      w.style.width = ri(60, 200) + "px";
      const dur = rand(2.6, 5.2).toFixed(2);
      const delay = rand(0, 4).toFixed(2);
      w.style.animation = `windSweep ${dur}s linear ${delay}s infinite`;
      wind.appendChild(w);
    }
    rootEl.appendChild(wind);

    // Lightning host (and full-page flash).
    const flash = document.createElement("div");
    flash.className = "storm-flash";
    rootEl.appendChild(flash);

    const ltn = document.createElement("div");
    ltn.className = "storm-lightning";
    rootEl.appendChild(ltn);

    // Ensure the page content is layered above atmosphere.
    [...rootEl.children].forEach(ch => {
      if (!ch.classList) return;
      if (ch.classList.contains("storm-clouds") ||
          ch.classList.contains("storm-rain")   ||
          ch.classList.contains("storm-wind")   ||
          ch.classList.contains("storm-lightning") ||
          ch.classList.contains("storm-flash")) return;
      // Wrap raw children in storm-content if they aren't already.
      if (!ch.classList.contains("storm-content")) {
        ch.style.position = ch.style.position || "relative";
        ch.style.zIndex = "10";
      }
    });

    // Ambient lightning loop — random strikes when intensity allows.
    if (opts.ambientLightning !== false) {
      scheduleAmbientStrike(rootEl);
    }
    return rootEl;
  }

  function scheduleAmbientStrike(rootEl) {
    const tick = () => {
      const intensity = rootEl.dataset.intensity || "normal";
      if (intensity === "off") {
        setTimeout(tick, 8000);
        return;
      }
      const meanGap = intensity === "high" ? 4500 : intensity === "low" ? 14000 : 8500;
      const gap = meanGap * rand(.55, 1.4);
      setTimeout(() => { triggerLightning(rootEl); tick(); }, gap);
    };
    tick();
  }

  // ---------------------------------------------------------
  // Lightning generator — branching tree, fresh every strike
  // ---------------------------------------------------------
  function buildBoltPath(x0, y0, x1, y1, segments) {
    const pts = [{ x: x0, y: y0 }];
    for (let i = 1; i < segments; i++) {
      const t = i / segments;
      const px = x0 + (x1 - x0) * t + rand(-30, 30);
      const py = y0 + (y1 - y0) * t + rand(-8, 8);
      pts.push({ x: px, y: py });
    }
    pts.push({ x: x1, y: y1 });
    let d = "M " + pts[0].x + " " + pts[0].y;
    for (let i = 1; i < pts.length; i++) d += " L " + pts[i].x + " " + pts[i].y;
    return { d, pts };
  }

  function triggerLightning(rootEl) {
    if (!rootEl) return;
    const host = rootEl.querySelector(".storm-lightning");
    if (!host) return;

    // Cap concurrent strikes — each cleans itself up.
    if (host.children.length >= 3) host.removeChild(host.firstChild);

    const w = rootEl.clientWidth;
    const h = rootEl.clientHeight || 600;

    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svg.setAttribute("preserveAspectRatio", "none");

    // glow filter
    const defs = document.createElementNS(NS, "defs");
    const fid = "glow_" + Math.random().toString(36).slice(2, 8);
    defs.innerHTML = `
      <filter id="${fid}" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="2.4" result="blur1"/>
        <feGaussianBlur stdDeviation="6"   result="blur2"/>
        <feMerge>
          <feMergeNode in="blur2"/>
          <feMergeNode in="blur1"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>`;
    svg.appendChild(defs);

    // main bolt
    const x0 = rand(w * .25, w * .75);
    const x1 = x0 + rand(-w * .15, w * .15);
    const main = buildBoltPath(x0, 0, x1, h * rand(.75, 1.0), ri(7, 11));

    const mainEl = document.createElementNS(NS, "path");
    mainEl.setAttribute("class", "bolt-main");
    mainEl.setAttribute("d", main.d);
    mainEl.setAttribute("stroke", "#e0e8ff");
    mainEl.setAttribute("stroke-width", "2.4");
    mainEl.setAttribute("fill", "none");
    mainEl.setAttribute("stroke-linecap", "round");
    mainEl.setAttribute("stroke-linejoin", "round");
    mainEl.setAttribute("filter", `url(#${fid})`);
    svg.appendChild(mainEl);

    // measure & animate dash
    const mainLen = mainEl.getTotalLength();
    mainEl.style.strokeDasharray  = mainLen;
    mainEl.style.setProperty("--len", mainLen);
    mainEl.style.strokeDashoffset = mainLen;

    // branches off ~half the inflection points
    const branchPaths = [];
    for (let i = 1; i < main.pts.length - 1; i++) {
      if (Math.random() < 0.55) {
        const a = main.pts[i];
        const next = main.pts[i + 1];
        const dx = next.x - a.x, dy = next.y - a.y;
        const len = Math.sqrt(dx*dx + dy*dy) * rand(.4, .65);
        const angDeg = (Math.random() < .5 ? 1 : -1) * rand(20, 38);
        const ang = Math.atan2(dy, dx) + angDeg * Math.PI / 180;
        const bx = a.x + Math.cos(ang) * len;
        const by = a.y + Math.sin(ang) * len;
        const branch = buildBoltPath(a.x, a.y, bx, by, ri(3, 6));
        const bEl = document.createElementNS(NS, "path");
        bEl.setAttribute("class", "bolt-branch");
        bEl.setAttribute("d", branch.d);
        bEl.setAttribute("stroke", "#c8d4ff");
        bEl.setAttribute("stroke-width", "1.4");
        bEl.setAttribute("fill", "none");
        bEl.setAttribute("stroke-linecap", "round");
        bEl.setAttribute("filter", `url(#${fid})`);
        bEl.setAttribute("opacity", ".85");
        svg.appendChild(bEl);
        branchPaths.push(bEl);
      }
    }
    branchPaths.forEach(b => {
      const L = b.getTotalLength();
      b.style.strokeDasharray  = L;
      b.style.setProperty("--len", L);
      b.style.strokeDashoffset = L;
    });

    host.appendChild(svg);

    // Fire flash + draw sequence
    const flashEl = rootEl.querySelector(".storm-flash");
    if (flashEl) {
      flashEl.classList.remove("fire");
      flashEl.offsetWidth;
      flashEl.classList.add("fire");
    }
    requestAnimationFrame(() => svg.classList.add("strike"));

    // self-cleanup
    setTimeout(() => { if (svg.parentNode) svg.parentNode.removeChild(svg); }, 560);
  }

  function setStormIntensity(rootEl, level) {
    if (!rootEl) return;
    rootEl.dataset.intensity = level || "normal";
  }

  // ---------------------------------------------------------
  // Spring physics on a model shape
  // ---------------------------------------------------------
  function mountSpringShape(wrapEl, opts) {
    if (!wrapEl) return null;
    opts = opts || {};
    const stiffness = opts.stiffness ?? 0.15;
    const damping   = opts.damping   ?? 0.75;

    if (!wrapEl.querySelector(".model-shape")) {
      wrapEl.innerHTML = `
        <div class="model-shape-aura"></div>
        <div class="model-shape"></div>
      `;
    }

    let target = 1.0;
    let scale  = 1.0;
    let vel    = 0;
    let raf    = null;
    let streak = 0;
    let verdict = "idle";
    wrapEl.dataset.verdict = verdict;

    function loop() {
      const force = (target - scale) * stiffness;
      vel = (vel + force) * damping;
      scale += vel;
      wrapEl.style.setProperty("--ms-scale", scale.toFixed(4));
      // Floor compresses inversely with scale (bigger shape ⇒ flatter shadow).
      const floor = 1 + (scale - 1) * 0.6;
      wrapEl.style.setProperty("--ms-floor", floor.toFixed(3));
      if (Math.abs(vel) > 0.0008 || Math.abs(target - scale) > 0.001) {
        raf = requestAnimationFrame(loop);
      } else {
        scale = target; vel = 0; raf = null;
        wrapEl.style.setProperty("--ms-scale", "1");
        wrapEl.style.setProperty("--ms-floor", "1");
      }
    }
    function bump(amount = 1.35) {
      // Snap to overshoot value, let the spring pull it back to 1.
      scale = amount;
      vel   = 0;
      target = 1.0;
      if (!raf) raf = requestAnimationFrame(loop);
    }

    function setVerdict(v) {
      verdict = v;
      wrapEl.dataset.verdict = v;
      bump(v === "bypassed" ? 1.45 : v === "blocked" ? 1.32 : 1.18);
    }

    function setStreak(n) {
      streak = Math.max(0, n | 0);
      wrapEl.dataset.streak = String(streak);
      const aura = wrapEl.querySelector(".model-shape-aura");
      if (!aura) return;
      aura.innerHTML = "";
      const tier = streak >= 10 ? 3 : streak >= 5 ? 2 : streak >= 3 ? 1 : 0;
      for (let i = 1; i <= tier; i++) {
        const r = document.createElement("div");
        r.className = "aura-ring r" + i;
        aura.appendChild(r);
      }
    }

    return { bump, setVerdict, setStreak,
             get streak() { return streak; },
             get verdict() { return verdict; } };
  }

  // ---------------------------------------------------------
  // Streak chip helper — for places where you show "🔥 4 in a row"
  // ---------------------------------------------------------
  function renderStreakChip(el, streak) {
    if (!el) return;
    if (!streak || streak < 1) { el.style.display = "none"; el.innerHTML = ""; return; }
    const tier = streak >= 10 ? 3 : streak >= 5 ? 2 : streak >= 3 ? 1 : 0;
    el.style.display = "inline-flex";
    el.dataset.tier = String(tier);
    el.innerHTML = `<span class="flame">🔥</span> ${streak} in a row`;
  }

  // =========================================================
  // TIER 2 — dial · shield · decision nodes · forecast ·
  //          split-view · click-burst · scroll reveal · morph
  // =========================================================

  // ---------- 24-vertex polygon builder for shape morph ----------
  // peaks=1 → circle; peaks=6 → hexagon; peaks=5 → 5-point star; etc.
  function buildPoly(peaks, peakR, valleyR, n) {
    n = n || 24;
    const pts = [];
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2 - Math.PI / 2;            // start at top
      const phase = (i / n) * peaks;
      const within = phase - Math.floor(phase);                  // 0..1
      const t = (Math.cos(within * Math.PI * 2) + 1) / 2;        // 1 at vertex, 0 at midpoint
      const r = valleyR + (peakR - valleyR) * t;
      const x = 50 + Math.cos(a) * r * 50;
      const y = 50 + Math.sin(a) * r * 50;
      pts.push(x.toFixed(1) + "% " + y.toFixed(1) + "%");
    }
    return "polygon(" + pts.join(", ") + ")";
  }
  const SHAPE_TIERS = {
    0: buildPoly(1, 0.95, 0.95),   // circle
    1: buildPoly(6, 0.96, 0.86),   // hexagon-ish (3+ streak)
    2: buildPoly(5, 0.98, 0.50),   // 5-point star (5+ streak)
    3: buildPoly(12, 1.0, 0.66),   // dodeca-star (10+ streak)
  };
  function streakTier(streak) {
    if (streak >= 10) return 3;
    if (streak >= 5)  return 2;
    if (streak >= 3)  return 1;
    return 0;
  }
  function applyShapeMorph(wrapEl, streak) {
    if (!wrapEl) return;
    const tier = streakTier(streak | 0);
    const shape = wrapEl.querySelector(".model-shape");
    if (!shape) return;
    wrapEl.dataset.morphing = "1";
    shape.style.clipPath = SHAPE_TIERS[tier];
    shape.style.webkitClipPath = SHAPE_TIERS[tier];
    // also drop the explicit border-radius so clip-path wins
    if (tier > 0) shape.style.borderRadius = "0";
    else          shape.style.borderRadius = "";
    setTimeout(() => { wrapEl.dataset.morphing = "0"; }, 650);
  }

  // ---------- storm dial (corner knob) ----------
  // Returns { setIntensityPct, getIntensityPct, releaseManual }.
  // Drag-rotates a brass knob; angle maps to 0..100% intensity.
  // Calling code can wire setStormIntensity(rootEl, ...) from getIntensityPct().
  function mountStormDial(opts) {
    opts = opts || {};
    const onChange = opts.onChange || function () {};
    const startPct = Math.max(0, Math.min(100, opts.initial ?? 50));
    const target   = opts.target || document.body;

    const el = document.createElement("div");
    el.className = "storm-dial";
    el.setAttribute("role", "slider");
    el.setAttribute("aria-label", "Storm intensity");
    el.setAttribute("aria-valuemin", "0");
    el.setAttribute("aria-valuemax", "100");
    el.innerHTML = `
      <div class="tick"></div>
      <div class="notch"></div>
      <svg class="bolt-icon" viewBox="0 0 20 20" aria-hidden="true">
        <path d="M11 1 L4 11 L9 11 L8 19 L16 8 L11 8 Z" fill="#ffe66c" stroke="#fff4c5" stroke-width="0.5"/>
      </svg>
      <div class="ring"></div>
      <div class="label">STORM</div>
    `;
    target.appendChild(el);

    // angle range: -135deg (0%) .. +135deg (100%), so 270deg total
    let pct = startPct;
    let dragging = false;
    let manual = false;
    const NOTCH_ZERO = -135; // degrees
    const NOTCH_SPAN = 270;

    function paint() {
      const ang = NOTCH_ZERO + (pct / 100) * NOTCH_SPAN;
      el.querySelector(".notch").style.transform = `translateX(-50%) rotate(${ang.toFixed(1)}deg)`;
      el.style.setProperty("--ring", (pct).toFixed(1) + "%");
      el.dataset.intensityPct = String(Math.round(pct));
      el.dataset.nearMax = pct >= 92 ? "1" : "0";
      el.setAttribute("aria-valuenow", String(Math.round(pct)));
    }
    paint();

    function pctFromEvent(e) {
      const rect = el.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const dx = e.clientX - cx;
      const dy = e.clientY - cy;
      let deg = Math.atan2(dy, dx) * 180 / Math.PI + 90;   // 0deg = up
      if (deg < NOTCH_ZERO) deg += 360;
      if (deg > NOTCH_ZERO + NOTCH_SPAN + 30) deg = NOTCH_ZERO + NOTCH_SPAN;
      if (deg < NOTCH_ZERO) deg = NOTCH_ZERO;
      let next = ((deg - NOTCH_ZERO) / NOTCH_SPAN) * 100;
      return Math.max(0, Math.min(100, next));
    }

    function emit() {
      onChange(pct, manual);
    }

    el.addEventListener("mousedown", (e) => {
      dragging = true; manual = true; e.preventDefault();
      pct = pctFromEvent(e); paint(); emit();
    });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      pct = pctFromEvent(e); paint(); emit();
    });
    window.addEventListener("mouseup", () => { dragging = false; });
    el.addEventListener("dblclick", () => {
      manual = false;
      onChange(pct, manual);   // signal handover back to auto
    });
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      manual = true;
      pct = Math.max(0, Math.min(100, pct + (e.deltaY < 0 ? 4 : -4)));
      paint(); emit();
    }, { passive: false });

    return {
      el,
      getIntensityPct: () => pct,
      setIntensityPct: (v) => { pct = Math.max(0, Math.min(100, v)); paint(); },
      isManual: () => manual,
      releaseManual: () => { manual = false; },
    };
  }

  // Translate 0..100% into the discrete intensity levels initStorm accepts.
  function intensityLevelFromPct(pct) {
    if (pct < 12) return "off";
    if (pct < 38) return "low";
    if (pct < 75) return "normal";
    return "high";
  }

  // ---------- safety shield ----------
  function mountSafetyShield(wrapEl) {
    if (!wrapEl) return null;
    let existing = wrapEl.querySelector(".safety-shield");
    if (!existing) {
      existing = document.createElement("div");
      existing.className = "safety-shield";
      existing.innerHTML = `
        <svg viewBox="0 0 100 110" preserveAspectRatio="xMidYMid meet">
          <defs>
            <linearGradient id="shieldFillGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stop-color="rgba(155,124,255,0.55)"/>
              <stop offset="100%" stop-color="rgba(60,30,150,0.85)"/>
            </linearGradient>
          </defs>
          <path class="shield-fill"
                d="M50 6 L88 18 L88 55 C88 80 70 100 50 105 C30 100 12 80 12 55 L12 18 Z"/>
          <path class="shield-outline"
                d="M50 6 L88 18 L88 55 C88 80 70 100 50 105 C30 100 12 80 12 55 L12 18 Z"/>
          <path class="shield-check"
                d="M32 56 L46 70 L70 42"/>
        </svg>
      `;
      wrapEl.appendChild(existing);
      const outline = existing.querySelector(".shield-outline");
      const peri = outline.getTotalLength ? outline.getTotalLength() : 360;
      existing.style.setProperty("--peri", peri.toFixed(1));
    }
    const shield = existing;

    function fire() {
      shield.classList.remove("active", "tracing", "flooding", "checking", "shrinking", "badge");
      // force reflow so we can re-fire animations
      void shield.offsetWidth;
      shield.classList.add("active", "tracing");
      setTimeout(() => shield.classList.add("flooding"), 580);
      setTimeout(() => shield.classList.add("checking"), 950);
      setTimeout(() => {
        // pulse once
        shield.classList.add("repulse");
        setTimeout(() => shield.classList.remove("repulse"), 500);
      }, 1500);
      setTimeout(() => shield.classList.add("shrinking"), 1900);
      setTimeout(() => {
        shield.classList.remove("shrinking", "tracing", "flooding", "checking");
        shield.classList.add("badge");
      }, 2550);
    }
    function clear() {
      shield.classList.remove("active", "tracing", "flooding", "checking", "shrinking", "badge");
    }
    return { el: shield, fire, clear };
  }

  // ---------- decision nodes ----------
  function mountDecisionNodes(rootEl, opts) {
    opts = opts || {};
    if (!rootEl) return null;
    let host = rootEl.querySelector(".decision-nodes");
    if (host) host.remove();
    host = document.createElement("div");
    host.className = "decision-nodes";
    rootEl.appendChild(host);

    const labels = opts.labels || ["INTENT", "POLICY", "REFUSE", "VERIFY", "EMIT"];
    const positions = opts.positions || [
      { x: 18, y: 22 }, { x: 42, y: 14 }, { x: 58, y: 26 },
      { x: 76, y: 18 }, { x: 90, y: 34 },
    ];
    const n = Math.min(labels.length, positions.length);
    const nodes = [];
    for (let i = 0; i < n; i++) {
      const node = document.createElement("div");
      node.className = "decision-node";
      node.style.left = positions[i].x + "%";
      node.style.top  = positions[i].y + "%";
      node.innerHTML = `<div class="label">${labels[i]}</div>`;
      host.appendChild(node);
      nodes.push(node);
    }

    function flashPath(indices) {
      // sequential flash along nodes — gives the "reasoning" feel
      indices.forEach((idx, i) => {
        setTimeout(() => {
          const n = nodes[idx];
          if (!n) return;
          n.classList.remove("flashing");
          void n.offsetWidth;
          n.classList.add("flashing");
        }, i * 80);
      });
    }
    function flashAll() {
      flashPath(nodes.map((_, i) => i));
    }
    function flashRandomPath() {
      // pick a path that hits 3 nodes left → right
      const left  = Math.floor(Math.random() * 2);
      const mid   = 2 + Math.floor(Math.random() * 1);
      const right = 3 + Math.floor(Math.random() * 2);
      flashPath([left, mid, right]);
    }

    return { el: host, nodes, flashPath, flashAll, flashRandomPath };
  }

  // Hook decision-node flash to ambient + manual lightning.
  function wireLightningToDecisions(rootEl, dec) {
    if (!rootEl || !dec) return;
    const originalTrigger = triggerLightning;
    rootEl.addEventListener("storm:strike", () => dec.flashRandomPath());
  }

  // Patch triggerLightning to dispatch a "storm:strike" CustomEvent so
  // decision-node listeners can react. Idempotent — only patches once.
  if (!global.__tokiTriggerPatched) {
    const inner = triggerLightning;
    triggerLightning = function (rootEl) {
      const r = inner(rootEl);
      try {
        if (rootEl && rootEl.dispatchEvent) {
          rootEl.dispatchEvent(new CustomEvent("storm:strike"));
        }
      } catch (_) { /* old browsers */ }
      return r;
    };
    global.__tokiTriggerPatched = true;
  }

  // ---------- weather forecast strip ----------
  // Forecast logic: based on recent verdicts, predict the next 5.
  // High pass rate → sun/cloud; mixed → cloud; many fails → rain/storm.
  const FORECAST_SVG = {
    sun:   '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4.5" fill="#fbbf24"/><g stroke="#fbbf24" stroke-width="1.6" stroke-linecap="round">' +
           [0, 45, 90, 135, 180, 225, 270, 315].map(a => {
             const r = a * Math.PI / 180;
             const x1 = 12 + Math.cos(r) * 7, y1 = 12 + Math.sin(r) * 7;
             const x2 = 12 + Math.cos(r) * 10, y2 = 12 + Math.sin(r) * 10;
             return `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"/>`;
           }).join("") + '</g></svg>',
    cloud: '<svg viewBox="0 0 24 24"><path fill="rgba(200,210,230,.85)" d="M7 16 Q3 16 3 12 Q3 9 6 8 Q6 5 10 5 Q14 5 15 8 Q19 8 19 12 Q19 16 16 16 Z"/></svg>',
    rain:  '<svg viewBox="0 0 24 24"><path fill="rgba(180,190,210,.85)" d="M7 13 Q3 13 3 10 Q3 7 6 6 Q6 4 10 4 Q14 4 15 7 Q19 7 19 10 Q19 13 16 13 Z"/>' +
           '<g stroke="#7eb6ff" stroke-width="1.4" stroke-linecap="round">' +
           '<line x1="8" y1="16" x2="6.5" y2="20"/><line x1="12" y1="17" x2="10.5" y2="21"/><line x1="16" y1="16" x2="14.5" y2="20"/>' +
           '</g></svg>',
    storm: '<svg viewBox="0 0 24 24"><path fill="rgba(120,110,160,.9)" d="M7 12 Q3 12 3 9 Q3 6 6 5 Q6 3 10 3 Q14 3 15 6 Q19 6 19 9 Q19 12 16 12 Z"/>' +
           '<path fill="#ffe66c" stroke="#fff4c5" stroke-width="0.4" d="M11 13 L7 19 L11 19 L9 22 L15 16 L11 16 Z"/></svg>',
  };
  function forecastIconFor(passRate) {
    if (passRate >= 0.92) return "sun";
    if (passRate >= 0.70) return "cloud";
    if (passRate >= 0.40) return "rain";
    return "storm";
  }
  function mountForecast(containerEl, n) {
    if (!containerEl) return null;
    n = n || 5;
    containerEl.classList.add("weather-forecast");
    containerEl.innerHTML = `
      <span class="label">Forecast</span>
      <div class="slots"></div>
    `;
    const slots = containerEl.querySelector(".slots");
    let history = [];   // 1=pass, 0=fail

    function predictNext() {
      // Use trailing average + tiny random walk. If no history, neutral.
      const tail = history.slice(-6);
      const base = tail.length ? tail.reduce((a, b) => a + b, 0) / tail.length : 0.7;
      const out = [];
      let trend = base;
      for (let i = 0; i < n; i++) {
        trend += (Math.random() - 0.5) * 0.12;
        trend = Math.max(0.05, Math.min(0.98, trend));
        out.push(trend);
      }
      return out;
    }
    function render() {
      const upcoming = predictNext();
      slots.innerHTML = "";
      upcoming.forEach((p, i) => {
        const key = forecastIconFor(p);
        const slot = document.createElement("div");
        slot.className = "forecast-slot " + key;
        slot.style.animationDelay = (i * 80) + "ms";
        slot.innerHTML = FORECAST_SVG[key] +
          '<div class="pct">' + Math.round(p * 100) + '%</div>';
        slots.appendChild(slot);
      });
    }
    function record(passed) {
      history.push(passed ? 1 : 0);
      if (history.length > 30) history = history.slice(-30);
      render();
    }
    render();
    return { record, render, el: containerEl };
  }

  // ---------- click-burst rain ----------
  function enableClickBurst(rootEl) {
    if (!rootEl) return;
    rootEl.addEventListener("click", (e) => {
      // Ignore clicks on interactive elements
      const t = e.target;
      if (!t) return;
      if (t.closest("button, a, input, select, textarea, .storm-dial")) return;

      const rect = rootEl.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const N = 18;
      for (let i = 0; i < N; i++) {
        const drop = document.createElement("div");
        drop.className = "burst-drop";
        const angle = (Math.random() * 0.7 - 0.35) + Math.PI * 0.5; // mostly downward
        const dist = 60 + Math.random() * 120;
        const dx = Math.cos(angle) * (Math.random() - 0.5) * 2 * dist;
        const dy = Math.sin(angle) * dist;
        const dur = 0.45 + Math.random() * 0.5;
        drop.style.left   = (x - 1) + "px";
        drop.style.top    = (y - 7) + "px";
        drop.style.height = (10 + Math.random() * 14) + "px";
        drop.style.transform = "translate(0,0) rotate(-15deg)";
        drop.style.opacity = 0.85;
        drop.style.transition = `transform ${dur}s ease-out, opacity ${dur}s ease-out`;
        rootEl.appendChild(drop);
        requestAnimationFrame(() => {
          drop.style.transform = `translate(${dx.toFixed(0)}px, ${dy.toFixed(0)}px) rotate(-15deg)`;
          drop.style.opacity = "0";
        });
        setTimeout(() => { if (drop.parentNode) drop.parentNode.removeChild(drop); }, dur * 1000 + 80);
      }
      // tiny lightning flicker at click point — extra sauce
      if (Math.random() < 0.25) triggerLightning(rootEl);
    });
  }

  // ---------- scroll reveal ----------
  function revealOnScroll(elements, opts) {
    opts = opts || {};
    const stagger = opts.stagger || 100;
    const direction = opts.direction || "auto";  // "auto" splits top→left bottom→right
    const seen = new WeakSet();
    elements.forEach((el, i) => {
      el.classList.add("reveal-init");
      const dir = direction === "auto"
        ? (i < Math.ceil(elements.length / 2) ? "from-left" : "from-right")
        : direction;
      if (dir === "from-right") el.classList.add("from-right");
      else if (dir === "from-up") el.classList.add("from-up");
    });
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        const el = entry.target;
        if (entry.isIntersecting) {
          const idx = elements.indexOf(el);
          setTimeout(() => {
            el.classList.add("in");
            if (seen.has(el)) el.classList.add("repulse");
            seen.add(el);
            setTimeout(() => el.classList.remove("repulse"), 720);
          }, idx >= 0 ? idx * stagger : 0);
        } else {
          el.classList.remove("in");
        }
      });
    }, { threshold: 0.18 });
    elements.forEach((el) => io.observe(el));
    return io;
  }

  // ---------- animated counter ----------
  function animateCounter(el, target, opts) {
    if (!el) return;
    opts = opts || {};
    const dur = opts.duration || 900;
    const from = opts.from || 0;
    const prefix = opts.prefix || "";
    const suffix = opts.suffix || "";
    const t0 = performance.now();
    function tick(now) {
      const t = Math.min(1, (now - t0) / dur);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - t, 3);
      const v = from + (target - from) * eased;
      el.textContent = prefix + Math.round(v) + suffix;
      if (t < 1) requestAnimationFrame(tick);
      else el.textContent = prefix + target + suffix;
    }
    requestAnimationFrame(tick);
  }

  // ---------- split-view ----------
  // containerEl is a div that becomes .split-view; opts.left/right name a
  // function(host, idx) that populates each pane's storm-content layer.
  function mountSplitView(containerEl, opts) {
    if (!containerEl) return null;
    opts = opts || {};
    containerEl.classList.add("split-view");
    containerEl.innerHTML = `
      <div class="storm-host" data-pane="left"><div class="storm-content split-pane-content"></div></div>
      <div class="split-divider"></div>
      <div class="storm-host" data-pane="right"><div class="storm-content split-pane-content"></div></div>
    `;
    const leftHost  = containerEl.querySelector('[data-pane="left"]');
    const rightHost = containerEl.querySelector('[data-pane="right"]');
    initStorm(leftHost,  { rainCount: 45, windCount: 4, ambientLightning: true });
    initStorm(rightHost, { rainCount: 45, windCount: 4, ambientLightning: true });
    setStormIntensity(leftHost,  opts.leftIntensity  || "normal");
    setStormIntensity(rightHost, opts.rightIntensity || "normal");
    if (opts.populateLeft)  opts.populateLeft(leftHost.querySelector(".split-pane-content"));
    if (opts.populateRight) opts.populateRight(rightHost.querySelector(".split-pane-content"));
    enableClickBurst(leftHost);
    enableClickBurst(rightHost);
    return { containerEl, leftHost, rightHost };
  }

  // ---------------------------------------------------------
  // Export
  // ---------------------------------------------------------
  global.initStorm          = initStorm;
  global.triggerLightning   = triggerLightning;
  global.setStormIntensity  = setStormIntensity;
  global.mountSpringShape   = mountSpringShape;
  global.renderStreakChip   = renderStreakChip;
  // Tier-2 additions
  global.applyShapeMorph    = applyShapeMorph;
  global.streakTier         = streakTier;
  global.buildPoly          = buildPoly;
  global.SHAPE_TIERS        = SHAPE_TIERS;
  global.mountStormDial     = mountStormDial;
  global.intensityLevelFromPct = intensityLevelFromPct;
  global.mountSafetyShield  = mountSafetyShield;
  global.mountDecisionNodes = mountDecisionNodes;
  global.mountForecast      = mountForecast;
  global.forecastIconFor    = forecastIconFor;
  global.enableClickBurst   = enableClickBurst;
  global.revealOnScroll     = revealOnScroll;
  global.animateCounter     = animateCounter;
  global.mountSplitView     = mountSplitView;
})(window);
