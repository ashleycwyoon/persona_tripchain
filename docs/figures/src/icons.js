// Minimal stroke icons, injected into <i data-ic="name" class="..."> placeholders.
const ICONS = {
  users: '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20v-1a5 5 0 0 1 5-5h3a5 5 0 0 1 5 5v1"/><circle cx="17" cy="7.5" r="2.8"/><path d="M18 13.5a4.5 4.5 0 0 1 3.5 4.4V20"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21v-1a6 6 0 0 1 6-6h4a6 6 0 0 1 6 6v1"/>',
  route: '<circle cx="6" cy="19" r="2.2"/><path d="M18 3.5c-2.2 0-4 1.7-4 3.9 0 2.9 4 6.6 4 6.6s4-3.7 4-6.6c0-2.2-1.8-3.9-4-3.9z"/><circle cx="18" cy="7.4" r="1.2"/><path d="M8.4 19H14a3 3 0 0 0 0-6h-4a3 3 0 0 1-3-3" stroke-dasharray="2 2.4"/>',
  chat: '<path d="M4 4.5h16v11.5H10l-5 4v-4H4z"/><path d="M8 8.5h8M8 12h5"/>',
  robot: '<rect x="4" y="8" width="16" height="12" rx="3"/><circle cx="9" cy="14" r="1.5"/><circle cx="15" cy="14" r="1.5"/><path d="M12 8V4.5"/><circle cx="12" cy="3.5" r="1"/><path d="M2 13v3M22 13v3"/>',
  doc: '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M9 8h6M9 12h6M9 16h4"/>',
  pin: '<path d="M12 21s-7-6.2-7-11.5A7 7 0 0 1 19 9.5C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  car: '<path d="M4 16.5h16v-4l-2.2-5H6.2L4 12.5z"/><path d="M4 12.5h16"/><circle cx="8" cy="17" r="1.8"/><circle cx="16" cy="17" r="1.8"/>',
  map: '<path d="M3 6l6-3 6 3 6-3v15l-6 3-6-3-6 3z"/><path d="M9 3v15M15 6v15"/>',
  trophy: '<path d="M8 4h8v5a4 4 0 0 1-8 0z"/><path d="M8 6H5a3 3 0 0 0 3 4M16 6h3a3 3 0 0 1-3 4"/><path d="M12 13v4M8 20.5h8M9.5 17h5v3.5h-5z"/>',
  network: '<circle cx="12" cy="4.5" r="2.2"/><circle cx="4.5" cy="18.5" r="2.2"/><circle cx="19.5" cy="18.5" r="2.2"/><circle cx="12" cy="12" r="2.4"/><circle cx="4.5" cy="9" r="1.6"/><circle cx="19.5" cy="9" r="1.6"/><path d="M12 6.7v2.9M10.2 13.6l-4.1 3.4M13.8 13.6l4.1 3.4M6 9.6l3.8 1.5M18 9.6l-3.8 1.5"/>',
  refresh: '<path d="M20 11a8 8 0 0 0-14.5-4.5L4 8"/><path d="M4 3v5h5"/><path d="M4 13a8 8 0 0 0 14.5 4.5L20 16"/><path d="M20 21v-5h-5"/>',
  check: '<rect x="3.5" y="3.5" width="17" height="17" rx="3.5"/><path d="M8 12.5l2.8 2.8L16.5 9"/>',
  chart: '<path d="M3 21h18"/><rect x="5" y="11" width="3" height="7"/><rect x="10.5" y="5" width="3" height="13"/><rect x="16" y="14" width="3" height="4"/>',
  db: '<ellipse cx="12" cy="5.5" rx="7" ry="2.5"/><path d="M5 5.5v13c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-13"/><path d="M5 12c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5"/>',
  split: '<circle cx="5" cy="12" r="2.2"/><circle cx="19" cy="5" r="2.2"/><circle cx="19" cy="12" r="2.2"/><circle cx="19" cy="19" r="2.2"/><path d="M7.2 12H17M7 11l10-5.2M7 13l10 5.2"/>',
};
document.querySelectorAll('i[data-ic]').forEach((el) => {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('class', 'ic ' + (el.className || ''));
  svg.innerHTML = ICONS[el.dataset.ic];
  if (el.style.color) svg.style.color = el.style.color;
  el.replaceWith(svg);
});

// Arrow overlay: coordinates come from the laid-out boxes, so arrows follow the layout.
const NS = 'http://www.w3.org/2000/svg';
function box(id) {
  const c = document.getElementById('canvas').getBoundingClientRect();
  const r = document.getElementById(id).getBoundingClientRect();
  const b = { l: r.left - c.left, r: r.right - c.left, t: r.top - c.top, b: r.bottom - c.top };
  b.cx = (b.l + b.r) / 2; b.cy = (b.t + b.b) / 2;
  return b;
}
function arrow(points, opts = {}) {
  const svg = document.getElementById('arrows');
  const p = document.createElementNS(NS, 'path');
  p.setAttribute('d', points.map((pt, i) => (i ? 'L' : 'M') + pt[0].toFixed(1) + ' ' + pt[1].toFixed(1)).join(' '));
  p.setAttribute('marker-end', 'url(#ah)');
  if (opts.dash) p.setAttribute('class', 'dash');
  if (opts.color) p.style.stroke = opts.color;
  svg.appendChild(p);
}
function label(x, y, text, opts = {}) {
  const svg = document.getElementById('arrows');
  const t = document.createElementNS(NS, 'text');
  t.setAttribute('x', x); t.setAttribute('y', y);
  t.setAttribute('text-anchor', opts.anchor || 'middle');
  t.textContent = text;
  if (opts.color) t.style.fill = opts.color;
  svg.appendChild(t);
  const bb = t.getBBox();
  const r = document.createElementNS(NS, 'rect');
  r.setAttribute('class', 'lbl');
  r.setAttribute('x', bb.x - 4); r.setAttribute('y', bb.y - 1);
  r.setAttribute('width', bb.width + 8); r.setAttribute('height', bb.height + 2);
  r.setAttribute('rx', 3);
  if (opts.bg) r.style.fill = opts.bg;
  svg.insertBefore(r, t);
}
function initArrows() {
  const svg = document.getElementById('arrows');
  svg.innerHTML = '<defs><marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#33475F"/></marker></defs>';
}
