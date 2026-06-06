import { displayToken } from "../core/format.js";

export function renderLineChart(svg, values, options) {
  if (!values.length) {
    svg.innerHTML = `<text x="24" y="${options.emptyY}" fill="#65717e" font-size="14">${options.emptyMessage}</text>`;
    return;
  }

  const { width, height, pad, stroke } = options;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-9);
  const points = values
    .map((v, i) => {
      const x = pad + (i / Math.max(values.length - 1, 1)) * (width - pad * 2);
      const y = height - pad - ((v - min) / span) * (height - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  svg.innerHTML = `
    <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#d8dee5" />
    <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#d8dee5" />
    <polyline points="${points}" fill="none" stroke="${stroke}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />
    <text x="${pad}" y="18" fill="#65717e" font-size="12">latest ${values.at(-1).toFixed(4)}</text>
    <text x="${width - 120}" y="18" fill="#65717e" font-size="12">min ${min.toFixed(4)}</text>
  `;
}

export function drawMatrix(canvas, matrix, options) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || canvas.width;
  const cssHeight = canvas.clientHeight || canvas.height;
  canvas.width = Math.floor(cssWidth * dpr);
  canvas.height = Math.floor(cssHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  const n = matrix.length;
  if (!n) {
    ctx.fillStyle = "#65717e";
    ctx.font = "14px system-ui";
    ctx.fillText(options.empty, 18, 36);
    return;
  }

  const labelSpace = options.compact ? 0 : 60;
  const pad = options.compact ? 12 : 22;
  const availableW = Math.max(cssWidth - labelSpace - pad * 2, 1);
  const availableH = Math.max(cssHeight - labelSpace - pad * 2, 1);
  const size = Math.min(availableW, availableH);
  const left = labelSpace + pad + Math.max((availableW - size) / 2, 0);
  const top = pad + Math.max((availableH - size) / 2, 0);
  const cell = size / n;
  const max = Math.max(...matrix.flat(), 1e-9);
  const [r, g, b] = options.color;

  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      const value = matrix[y][x] / max;
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.08 + value * 0.88})`;
      ctx.fillRect(left + x * cell, top + y * cell, Math.ceil(cell), Math.ceil(cell));
    }
  }

  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1;
  for (let i = 0; i <= n; i += 1) {
    const col = left + i * cell;
    ctx.beginPath();
    ctx.moveTo(col, top);
    ctx.lineTo(col, top + size);
    ctx.stroke();
    const row = top + i * cell;
    ctx.beginPath();
    ctx.moveTo(left, row);
    ctx.lineTo(left + size, row);
    ctx.stroke();
  }

  if (options.labels && !options.compact) {
    ctx.fillStyle = "#65717e";
    ctx.font = "11px ui-monospace, monospace";
    options.labels.forEach((token, i) => {
      const label = displayToken(token);
      ctx.fillText(label, 8, top + i * cell + cell * 0.62);
      ctx.save();
      ctx.translate(left + i * cell + cell * 0.32, top + size + 38);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(label, 0, 0);
      ctx.restore();
    });
  }
}

export function drawPixelImage(canvas, pixels) {
  const size = pixels.length;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(size, size);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const value = Math.max(0, Math.min(255, Math.round(pixels[y][x] * 255)));
      const offset = (y * size + x) * 4;
      image.data[offset] = value;
      image.data[offset + 1] = value;
      image.data[offset + 2] = value;
      image.data[offset + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
}
