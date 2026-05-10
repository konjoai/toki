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

  // ---------------------------------------------------------
  // Export
  // ---------------------------------------------------------
  global.initStorm         = initStorm;
  global.triggerLightning  = triggerLightning;
  global.setStormIntensity = setStormIntensity;
  global.mountSpringShape  = mountSpringShape;
  global.renderStreakChip  = renderStreakChip;
})(window);
