(() => {
  const canvas = document.getElementById("showcase-canvas");
  const video = document.getElementById("showcase-video");
  const prompt = document.getElementById("showcase-prompt");
  const frameLabel = document.getElementById("showcase-frame");
  const tabs = [...document.querySelectorAll(".showcase-tab")];
  if (!canvas || !video || !prompt || !frameLabel || !tabs.length) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let sectionVisible = false;
  let resumeOnView = !reducedMotion;
  const state = {
    examples: [], current: null, azimuth: -0.6, elevation: 0.54,
    width: 0, height: 0, dragging: false, pointerX: 0, pointerY: 0, animation: 0,
  };
  const add = (a, b) => a.map((v, i) => v + b[i]);
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const mul = (a, scalar) => a.map((v) => v * scalar);
  const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const unit = (v) => mul(v, 1 / (Math.hypot(...v) || 1));
  const mix = (a, b, t) => a.map((v, i) => v * (1 - t) + b[i] * t);

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    state.width = rect.width;
    state.height = rect.height;
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    draw();
  }

  function line(a, b, color, width = 1) {
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  }

  function dot2d(point, radius, fill, stroke = "#fff") {
    ctx.beginPath();
    ctx.arc(point[0], point[1], radius, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  function drawFrustum(position, right, up, project, extent) {
    const x = unit(right);
    const y = unit(sub(up, mul(x, dot(up, x))));
    const forward = mul(cross(x, y), -1); // OpenGL camera looks down local -Z.
    const depth = extent * 0.14;
    const center = add(position, mul(forward, depth));
    const halfRight = mul(x, depth * 0.72);
    const halfUp = mul(y, depth * 0.48);
    const corners = [
      add(add(center, halfRight), halfUp),
      add(sub(center, halfRight), halfUp),
      sub(sub(center, halfRight), halfUp),
      add(sub(center, halfUp), halfRight),
    ].map(project);
    const eye = project(position);
    ctx.beginPath();
    corners.forEach((corner, index) => {
      if (index === 0) ctx.moveTo(...corner);
      else ctx.lineTo(...corner);
    });
    ctx.closePath();
    ctx.fillStyle = "rgba(40,112,192,.11)";
    ctx.fill();
    ctx.strokeStyle = "#2870c0";
    ctx.lineWidth = 1.6;
    ctx.stroke();
    corners.forEach((corner) => line(eye, corner, "#2870c0", 1.3));
    line(eye, project(add(position, mul(forward, depth * 1.7))), "#e1794f", 2);
  }

  function draw() {
    if (!state.width || !state.height) return;
    const width = state.width;
    const height = state.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#fafbfd";
    ctx.fillRect(0, 0, width, height);
    const example = state.current;
    if (!example) return;

    const frames = example.frames;
    const positions = frames.map((frame) => frame.slice(0, 3));
    const mins = [0, 1, 2].map((axis) => Math.min(...positions.map((p) => p[axis])));
    const maxs = [0, 1, 2].map((axis) => Math.max(...positions.map((p) => p[axis])));
    const center = mins.map((value, axis) => (value + maxs[axis]) / 2);
    const extent = Math.max(...mins.map((value, axis) => maxs[axis] - value), 0.7);
    const az = state.azimuth;
    const el = state.elevation;
    const eye = [Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)];
    const viewRight = unit(cross([0, 1, 0], eye));
    const viewUp = cross(eye, viewRight);
    const scale = Math.min(width, height) * 0.67 / extent;
    const project = (point) => {
      const relative = sub(point, center);
      const perspective = 5 * extent / (5 * extent - dot(relative, eye));
      return [
        width / 2 + dot(relative, viewRight) * scale * perspective,
        height / 2 - dot(relative, viewUp) * scale * perspective,
      ];
    };

    const ground = mins[1] - extent * 0.12;
    for (let step = -2; step <= 2; step++) {
      const offset = step * extent / 4;
      line(project([center[0] + offset, ground, center[2] - extent * 0.6]),
        project([center[0] + offset, ground, center[2] + extent * 0.6]), "#e2e9f1");
      line(project([center[0] - extent * 0.6, ground, center[2] + offset]),
        project([center[0] + extent * 0.6, ground, center[2] + offset]), "#e2e9f1");
    }

    const origin = [0, 0, 0];
    const axes = [
      { end: [extent * 0.3, 0, 0], color: "#c57469", label: "+X" },
      { end: [0, extent * 0.3, 0], color: "#63967c", label: "+Y" },
      { end: [0, 0, -extent * 0.3], color: "#7790bf", label: "−Z" },
    ];
    ctx.font = "600 10px Lato, sans-serif";
    axes.forEach((axis) => {
      const end = project(axis.end);
      line(project(origin), end, axis.color, 1.4);
      ctx.fillStyle = axis.color;
      ctx.fillText(axis.label, end[0] + 4, end[1] - 4);
    });

    ctx.beginPath();
    positions.forEach((point, index) => {
      const projected = project(point);
      if (index === 0) ctx.moveTo(...projected);
      else ctx.lineTo(...projected);
    });
    ctx.strokeStyle = "#bccbdd";
    ctx.lineWidth = 2;
    ctx.stroke();

    const duration = video.duration;
    const progress = Number.isFinite(duration) && duration > 0
      ? Math.max(0, Math.min(1, video.currentTime / duration)) : 0;
    const cursor = progress * (frames.length - 1);
    const index = Math.min(frames.length - 1, Math.floor(cursor));
    const alpha = cursor - index;
    const next = frames[Math.min(index + 1, frames.length - 1)];
    const position = mix(frames[index].slice(0, 3), next.slice(0, 3), alpha);
    const right = mix(frames[index].slice(3, 6), next.slice(3, 6), alpha);
    const up = mix(frames[index].slice(6, 9), next.slice(6, 9), alpha);
    ctx.beginPath();
    ctx.moveTo(...project(positions[0]));
    for (let i = 1; i <= index; i++) ctx.lineTo(...project(positions[i]));
    ctx.lineTo(...project(position));
    ctx.strokeStyle = "#2870c0";
    ctx.lineWidth = 3.2;
    ctx.stroke();

    dot2d(project(positions[0]), 4, "#5a9b7d");
    dot2d(project(positions[positions.length - 1]), 4, "#8b9aaf");
    drawFrustum(position, right, up, project, extent);
    dot2d(project(position), 5, "#e1794f");
    frameLabel.textContent = String(Math.round(cursor) + 1).padStart(2, "0") + " / " + frames.length;
  }

  function animate() {
    cancelAnimationFrame(state.animation);
    if (video.paused) return;
    const tick = () => {
      draw();
      if (!video.paused) state.animation = requestAnimationFrame(tick);
    };
    state.animation = requestAnimationFrame(tick);
  }

  function selectExample(id) {
    const example = state.examples.find((item) => item.id === id);
    if (!example) return;
    state.current = example;
    prompt.textContent = example.prompt;
    canvas.setAttribute("aria-label", "Animated three-dimensional camera path for " + example.title);
    tabs.forEach((tab) => {
      const selected = tab.dataset.example === id;
      tab.classList.toggle("is-active", selected);
      tab.setAttribute("aria-selected", String(selected));
    });
    video.pause();
    video.src = "static/videos/" + id + ".mp4";
    video.load();
    draw();
    if (!reducedMotion) {
      if (sectionVisible) video.play().catch(() => {});
      else resumeOnView = true;
    }
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectExample(tab.dataset.example));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const next = tabs[(index + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      next.focus();
      selectExample(next.dataset.example);
    });
  });
  video.addEventListener("play", animate);
  video.addEventListener("pause", () => {
    cancelAnimationFrame(state.animation);
    draw();
  });
  for (const event of ["loadedmetadata", "seeked", "timeupdate"]) video.addEventListener(event, draw);
  canvas.addEventListener("pointerdown", (event) => {
    state.dragging = true;
    state.pointerX = event.clientX;
    state.pointerY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!state.dragging) return;
    state.azimuth += (event.clientX - state.pointerX) * 0.008;
    state.elevation = Math.max(0.12, Math.min(1.32, state.elevation + (event.clientY - state.pointerY) * 0.008));
    state.pointerX = event.clientX;
    state.pointerY = event.clientY;
    draw();
  });
  for (const event of ["pointerup", "pointercancel"]) canvas.addEventListener(event, () => { state.dragging = false; });
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(([entry]) => {
      sectionVisible = entry.isIntersecting;
      if (!sectionVisible && !video.paused) {
        resumeOnView = true;
        video.pause();
      } else if (sectionVisible && resumeOnView && !reducedMotion) {
        resumeOnView = false;
        video.play().catch(() => {});
      }
    }, { threshold: 0.2 }).observe(document.getElementById("showcase"));
  } else {
    sectionVisible = true;
  }
  if ("ResizeObserver" in window) new ResizeObserver(resize).observe(canvas);
  else window.addEventListener("resize", resize);
  resize();

  fetch("static/data/showcase.json")
    .then((response) => {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.json();
    })
    .then((data) => {
      state.examples = data.examples.filter((example) =>
        Array.isArray(example.frames) && example.frames.length > 1 &&
        example.frames.every((frame) => frame.length === 9 && frame.every(Number.isFinite)));
      selectExample("000952");
    })
    .catch(() => {
      ctx.fillStyle = "#667080";
      ctx.font = "14px Lato, sans-serif";
      ctx.fillText("Trajectory preview unavailable", 18, 36);
    });
})();
