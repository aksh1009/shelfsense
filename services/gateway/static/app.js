// Shelf product grouping - web interface.
// Uploads the photo to POST /api/v1/analyze, then draws the returned boxes on a canvas so a group can be
// highlighted. The server also saves its own visualization (shown under "Saved visualization").
(() => {
  const $ = (id) => document.getElementById(id);
  const state = { file: null, imageURL: null, result: null, image: null, active: null, hover: null };
  const STAGES = [
    ["decode", "Read image", "#9aa4aa"],
    ["detect", "Detect products", "#0082c8"],
    ["group", "Group products", "#3cb44b"],
    ["visualize", "Draw result", "#f58230"],
    ["save", "Save", "#b8c0c5"],
  ];

  // ------------------------------------------------------------------ service status
  async function refreshHealth() {
    const items = document.querySelectorAll("#services li");
    try {
      const body = await (await fetch("/api/v1/health")).json();
      items.forEach((li) => {
        const s = li.dataset.service;
        const ok = s === "gateway" || body.services?.[s]?.status === "ok";
        li.classList.toggle("ok", ok); li.classList.toggle("bad", !ok);
        li.title = ok ? "Running" : (body.services?.[s]?.error || "Not reachable");
      });
    } catch {
      items.forEach((li) => { li.classList.remove("ok"); li.classList.add("bad"); });
    }
  }

  // ------------------------------------------------------------------ choosing a photo
  function setFile(file, label) {
    state.file = file;
    if (state.imageURL) URL.revokeObjectURL(state.imageURL);
    state.imageURL = URL.createObjectURL(file);
    $("dz-title").textContent = label || file.name;
    $("dz-sub").textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB - press Analyze photo`;
    $("dropzone").classList.add("has-file");
    $("analyze-btn").disabled = false;
    setStatus("");
  }

  $("file-input").addEventListener("change", (e) => {
    if (e.target.files[0]) { clearSampleSelection(); setFile(e.target.files[0]); }
  });
  const dz = $("dropzone");
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) { clearSampleSelection(); setFile(f); }
    else setStatus("That file isn't an image. Drop a JPEG, PNG or WebP photo.", "error");
  });
  const clearSampleSelection = () => document.querySelectorAll(".sample").forEach((b) => b.setAttribute("aria-pressed", "false"));

  async function loadSamples() {
    try {
      const { samples } = await (await fetch("/api/v1/samples")).json();
      samples.slice(0, 20).forEach((s) => {
        const b = document.createElement("button");
        b.className = "sample"; b.type = "button"; b.setAttribute("role", "listitem");
        b.setAttribute("aria-pressed", "false"); b.setAttribute("aria-label", `Sample ${s.name}`); b.title = s.name;
        b.style.backgroundImage = `url("${s.url}")`;
        b.addEventListener("click", async () => {
          clearSampleSelection(); b.setAttribute("aria-pressed", "true");
          setStatus("Loading sample...", "busy");
          const blob = await (await fetch(s.url)).blob();
          setFile(new File([blob], s.name, { type: blob.type || "image/jpeg" }), s.name);
        });
        $("sample-strip").appendChild(b);
      });
      if (!samples.length) document.querySelector(".samples").hidden = true;
    } catch { document.querySelector(".samples").hidden = true; }
  }

  // ------------------------------------------------------------------ analyze
  function setStatus(text, kind) {
    $("status").textContent = text;
    $("status").className = "status" + (kind ? ` ${kind}` : "");
  }

  $("analyze-btn").addEventListener("click", async () => {
    if (!state.file) return;
    const btn = $("analyze-btn");
    btn.disabled = true;
    setStatus("Detecting and grouping products...", "busy");
    const form = new FormData();
    form.append("image", state.file, state.file.name);
    const t0 = performance.now();
    try {
      const r = await fetch("/api/v1/analyze", { method: "POST", body: form });
      const body = await r.json().catch(() => ({}));
      if (!r.ok || body.status !== "success") throw new Error(body.error?.message || `The server returned HTTP ${r.status}.`);
      state.result = body;
      await showResult(body);
      setStatus(`Done in ${((performance.now() - t0) / 1000).toFixed(1)} s (including upload).`);
      $("dz-sub").textContent = "Analyzed. Choose another photo to compare.";
    } catch (e) {
      setStatus(e.message, "error");
    } finally {
      btn.disabled = false;
      refreshHealth();
    }
  });

  // ------------------------------------------------------------------ result
  async function showResult(res) {
    state.active = null; state.hover = null;
    state.image = await loadImage(state.imageURL);
    $("results").hidden = false;
    $("sum-products").textContent = res.summary.num_products;
    $("sum-groups").textContent = res.summary.num_groups;
    $("sum-singles").textContent = res.summary.num_singletons;
    $("sum-time").textContent = formatMs(res.timings_ms.total);
    renderTiming(res.timings_ms);
    renderTags(res);
    $("dl-vis").href = res.visualization.url;
    $("dl-vis").setAttribute("download", `${res.request_id}_visualization.jpg`);
    $("dl-json").href = res.result_url;
    $("dl-json").setAttribute("download", `${res.request_id}_result.json`);
    $("saved-img").src = res.visualization.url;
    $("json-out").textContent = JSON.stringify(res, null, 2);
    ["json-panel", "viewer-tabs"].forEach((id) => { $(id).hidden = false; });
    $("empty").hidden = true;
    setView("interactive");
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Could not display the photo."));
      img.src = src;
    });
  }

  const formatMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);

  function renderTiming(t) {
    const total = STAGES.reduce((s, [k]) => s + (t[k] || 0), 0) || 1;
    $("timing-bar").innerHTML = STAGES.map(([k, , c]) =>
      `<span style="width:${(100 * (t[k] || 0)) / total}%;background:${c}" title="${k}"></span>`).join("");
    $("timing-legend").innerHTML = STAGES.map(([k, label, c]) =>
      `<li><i style="background:${c}"></i>${label} ${formatMs(t[k] || 0)}</li>`).join("");
  }

  function renderTags(res) {
    const ul = $("tags");
    ul.innerHTML = "";
    res.groups.filter((g) => !g.is_singleton).forEach((g) => ul.appendChild(makeTag(g.group_id, `${g.num_members} products`, g.color, false)));
    const singles = res.groups.filter((g) => g.is_singleton).length;
    if (singles) ul.appendChild(makeTag("__singles", `${singles} single`, "#a0a0a0", true, "Singles"));
  }

  function makeTag(id, sub, color, single, title) {
    const li = document.createElement("li"), b = document.createElement("button");
    b.type = "button"; b.className = "tag" + (single ? " single" : "");
    b.style.setProperty("--c", color);
    b.setAttribute("aria-pressed", "false");
    b.innerHTML = `<strong>${title || id}</strong><small>${sub}</small>`;
    const hover = (v) => () => { state.hover = v; draw(); };
    b.addEventListener("mouseenter", hover(id)); b.addEventListener("mouseleave", hover(null));
    b.addEventListener("focus", hover(id)); b.addEventListener("blur", hover(null));
    b.addEventListener("click", () => {
      state.active = state.active === id ? null : id;
      document.querySelectorAll(".tag").forEach((t) => t.setAttribute("aria-pressed", "false"));
      if (state.active) b.setAttribute("aria-pressed", "true");
      setView("interactive"); draw();
    });
    li.appendChild(b);
    return li;
  }

  // ------------------------------------------------------------------ drawing
  function setView(view) {
    document.querySelectorAll("#viewer-tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.view === view)));
    $("canvas").hidden = view !== "interactive";
    $("saved-img").hidden = view !== "saved";
    if (view === "interactive") draw();
  }
  document.querySelectorAll("#viewer-tabs button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));

  function draw() {
    const res = state.result, img = state.image;
    if (!res || !img) return;
    const canvas = $("canvas");
    const scale = Math.min(img.naturalWidth, 2200) / img.naturalWidth;
    canvas.width = Math.round(img.naturalWidth * scale);
    canvas.height = Math.round(img.naturalHeight * scale);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const focus = state.hover || state.active;
    const colorOf = Object.fromEntries(res.groups.map((g) => [g.group_id, g.color]));
    const single = new Set(res.groups.filter((g) => g.is_singleton).map((g) => g.group_id));
    const inFocus = (gid) => !focus || gid === focus || (focus === "__singles" && single.has(gid));
    if (focus) {
      ctx.fillStyle = "rgba(20, 28, 33, 0.55)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      res.detections.forEach((d) => {
        if (!inFocus(d.group_id)) return;
        const { x1, y1, x2, y2 } = d.bbox;
        ctx.drawImage(img, x1, y1, x2 - x1, y2 - y1, x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
      });
    }
    const lw = Math.max(2, canvas.width / 450);
    const fontPx = Math.max(11, Math.round(canvas.width / 75));
    ctx.font = `700 ${fontPx}px Archivo, Arial, sans-serif`;
    res.detections.forEach((d) => {
      if (focus && !inFocus(d.group_id)) return;
      const { x1, y1, x2, y2 } = d.bbox;
      const isSingle = single.has(d.group_id);
      ctx.strokeStyle = colorOf[d.group_id];
      ctx.lineWidth = isSingle ? lw / 2 : lw;
      ctx.strokeRect(x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
      if (!isSingle && (x2 - x1) * scale > canvas.width / 40) {
        const w = ctx.measureText(d.group_id).width + lw * 2 + 4;
        ctx.fillStyle = colorOf[d.group_id];
        ctx.fillRect(x1 * scale, y1 * scale, w, fontPx + lw + 4);
        ctx.fillStyle = luminance(colorOf[d.group_id]) > 140 ? "#000" : "#fff";
        ctx.fillText(d.group_id, x1 * scale + lw + 2, y1 * scale + fontPx + 1);
      }
    });
  }

  function luminance(hex) {
    const n = parseInt(hex.slice(1), 16);
    return 0.299 * (n >> 16) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255);
  }

  loadSamples();
  refreshHealth();
  setInterval(refreshHealth, 15000);
})();
