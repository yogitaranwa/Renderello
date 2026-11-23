// Preview object URLs outlive removed DOM nodes unless explicitly revoked.
const sourcePreviewUrls = new WeakMap();
export function clearResults(resultsBox) {
  for (const url of sourcePreviewUrls.get(resultsBox) || []) URL.revokeObjectURL(url);
  sourcePreviewUrls.delete(resultsBox);
  resultsBox.replaceChildren();
}

// Results: run summary, honesty lines, and a comparison gallery. Every
// variant card can show the generated shot beside the uploaded
// source photo (client-side object URL; sources are not stored server-side).
// QA labels stay verbatim: unscored variants say eye review is required and
// publish-ready is no.
export function renderResults(manifest, resultsBox, sourceFiles = [], opts = {}) {
  let host = resultsBox;
  if (opts.append) {
    host = document.createElement('section');
    host.className = 'run-group';
    if (opts.title) {
      const h = document.createElement('h3');
      h.className = 'run-group-title';
      h.textContent = opts.title;
      host.appendChild(h);
    }
    resultsBox.appendChild(host);
  } else {
    clearResults(resultsBox);
  }

  // Run summary strip
  const summary = document.createElement('div');
  summary.className = 'run-summary';
  const tags = [
    ['run', manifest.run_dir],
    ['mode', manifest.mode],
    ['engine', manifest.provider_used || manifest.provider_requested],
    ['template', manifest.preset],
    ['shape', manifest.aspect],
    ['time', (manifest.elapsed_s != null ? manifest.elapsed_s + 's' : null)],
  ];
  for (const [k, v] of tags) {
    if (!v) continue;
    const t = document.createElement('span');
    t.className = 'tag';
    t.innerHTML = k + ' <b></b>';
    t.querySelector('b').textContent = String(v);
    summary.appendChild(t);
  }
  if (manifest.credits_note) {
    const t = document.createElement('span');
    t.className = 'tag';
    t.textContent = manifest.credits_note;
    summary.appendChild(t);
  }
  const spacer = document.createElement('span');
  spacer.className = 'spacer';
  summary.appendChild(spacer);
  if (manifest.contact_sheet) {
    const a = document.createElement('a');
    a.className = 'btn-mini';
    a.href = '/api/runs/' + manifest.run_dir + '/' + manifest.contact_sheet;
    a.target = '_blank';
    a.rel = 'noopener';
    a.textContent = 'Contact sheet';
    summary.appendChild(a);
  }
  host.appendChild(summary);

  // Honesty lines: cutout backends and warnings are always visible.
  const cuts = (manifest.products || []).filter(p => p.cutout_backend);
  if (cuts.length) {
    const info = document.createElement('p');
    info.className = 'notice-line info';
    info.textContent = 'Cutout backend: ' +
      cuts.map(c => c.file + ' -> ' + c.cutout_backend).join(', ');
    host.appendChild(info);
  }
  for (const w of manifest.warnings || []) {
    const p = document.createElement('p');
    p.className = 'notice-line warn';
    p.textContent = w;
    host.appendChild(p);
  }

  // A group composite has no one-to-one mapping to its first input photo.
  // Do not suggest that comparing only that photo checks every product.
  const hasImage = manifest.variants.some(variant => !variant.error && variant.image);
  const srcUrl = sourceFiles.length === 1 && hasImage ? URL.createObjectURL(sourceFiles[0]) : null;
  if (srcUrl) {
    if (!sourcePreviewUrls.has(resultsBox)) sourcePreviewUrls.set(resultsBox, new Set());
    sourcePreviewUrls.get(resultsBox).add(srcUrl);
  }
  if (sourceFiles.length > 1) {
    const notice = document.createElement('p');
    notice.className = 'notice-line info';
    notice.textContent = 'Multiple source photos: review every original above. A single-source comparison overlay is not available for this group.';
    host.appendChild(notice);
  }

  const grid = document.createElement('div');
  grid.className = 'variant-grid';
  host.appendChild(grid);

  for (const v of manifest.variants) {
    const card = document.createElement('div');
    card.className = 'vcard';
    if (srcUrl) card.classList.add('vcard-comparison');
    if (v.error) {
      const p = document.createElement('p');
      p.className = 'vcard-fail';
      p.textContent = 'Variant failed: ' + v.error;
      card.appendChild(p);
      grid.appendChild(card);
      continue;
    }
    const imgUrl = '/api/runs/' + manifest.run_dir + '/' + v.image;

    const media = document.createElement('div');
    media.className = 'vcard-media';
    if (srcUrl) {
      media.appendChild(makeCompare(srcUrl, imgUrl));
    } else {
      const img = document.createElement('img');
      img.src = imgUrl;
      img.alt = 'Generated variant ' + (v.index + 1);
      media.appendChild(img);
    }
    card.appendChild(media);

    const body = document.createElement('div');
    body.className = 'vcard-body';
    const top = document.createElement('div');
    top.className = 'vcard-top';
    const engine = document.createElement('span');
    engine.textContent = (v.provider || v.engine || 'engine') + ' · seed ' + v.seed;
    top.appendChild(engine);
    body.appendChild(top);

    // QA badge: scored or explicitly unscored (eye review required).
    if (v.qa) {
      const q = v.qa;
      const unscored = !q.metrics || q.metrics.ssim_product == null ||
        (q.critical_errors || []).some(e => String(e).includes('unscored'));
      const badge = document.createElement('span');
      if (unscored) {
        badge.className = 'qa-badge review';
        badge.textContent = 'Eye review required · not auto-scored · publish-ready: no';
      } else {
        badge.className = 'qa-badge scored';
        badge.textContent = 'Fidelity ' + q.fidelity_score + '/5 · publish-ready: ' +
          (q.publish_ready ? 'yes' : 'no');
      }
      body.appendChild(badge);
      const ce = q.critical_alteration || (q.critical_errors || []).join('; ');
      if (ce) {
        const d = document.createElement('p');
        d.className = 'qa-detail';
        const s = document.createElement('span');
        s.className = 'bad';
        s.textContent = ce;
        d.appendChild(s);
        body.appendChild(d);
      }
    }

    // Per-variant warnings include scene intrusion after the call limit.
    // Keep them visible on the card; manifest-only storage hides the reason
    // a result still needs review. textContent keeps provider text inert.
    for (const warning of v.warnings || []) {
      const notice = document.createElement('p');
      notice.className = 'notice-line warn variant-warning';
      notice.textContent = String(warning);
      body.appendChild(notice);
    }

    const actions = document.createElement('div');
    actions.className = 'vcard-actions';
    const open = document.createElement('a');
    open.className = 'btn-mini';
    open.href = imgUrl;
    open.target = '_blank';
    open.rel = 'noopener';
    open.textContent = 'Open full size';
    actions.appendChild(open);
    const dl = document.createElement('a');
    dl.className = 'btn-mini';
    dl.href = imgUrl;
    dl.setAttribute('download', v.image);
    dl.textContent = 'Download';
    actions.appendChild(dl);
    body.appendChild(actions);

    card.appendChild(body);
    grid.appendChild(card);
  }
}

// Equal image aspect ratios do not establish aligned product placement.
// Always show both complete frames. A default 50% wipe can hide the entire
// generated product when the compositor moves it to the other side.
function makeCompare(srcUrl, genUrl) {
  const wrap = document.createElement('div');
  wrap.className = 'compare';
  makePair(wrap, srcUrl, genUrl, 1);
  srcImg_probe(srcUrl, genUrl, (srcAspect, genAspect) => {
    wrap.style.aspectRatio = String(2 * Math.min(srcAspect, genAspect));
  });
  return wrap;
}

export function renderSkeletons(resultsBox, count) {
  clearResults(resultsBox);
  const grid = document.createElement('div');
  grid.className = 'variant-grid';
  for (let i = 0; i < Math.max(1, Math.min(count, 8)); i++) {
    const s = document.createElement('div');
    s.className = 'skel';
    s.innerHTML = '<div class="skel-img"></div><div class="skel-line"></div><div class="skel-line short"></div>';
    grid.appendChild(s);
  }
  resultsBox.appendChild(grid);
}

export function renderError(resultsBox, message, opts = {}) {
  let host = resultsBox;
  if (opts.append) {
    host = document.createElement('section');
    host.className = 'run-group';
    if (opts.title) {
      const h = document.createElement('h3');
      h.className = 'run-group-title';
      h.textContent = opts.title;
      host.appendChild(h);
    }
    resultsBox.appendChild(host);
  } else {
    clearResults(resultsBox);
  }
  const p = document.createElement('p');
  p.className = 'notice-line fail';
  p.textContent = message;
  host.appendChild(p);
}

// Probe both natural sizes, then call back with their aspects.
function srcImg_probe(srcUrl, genUrl, cb) {
  let a = null, b = null;
  const done = () => { if (a && b) cb(a, b); };
  const ia = new Image();
  ia.onload = () => { a = ia.naturalWidth / ia.naturalHeight; done(); };
  ia.src = srcUrl;
  const ib = new Image();
  ib.onload = () => { b = ib.naturalWidth / ib.naturalHeight; done(); };
  ib.src = genUrl;
}

// Side-by-side fitted panels keep both complete frames visible.
// Captions are outside the image area, never over the product.
function makePair(wrap, srcUrl, genUrl, minAspect) {
  wrap.classList.add('pair');
  wrap.style.aspectRatio = String(2 * minAspect);
  wrap.innerHTML =
    '<figure class="pair-cell"><div class="pair-frame"><img alt="Source product photo"></div><figcaption class="cmp-tag src">Source</figcaption></figure>' +
    '<figure class="pair-cell"><div class="pair-frame"><img alt="Generated variant"></div><figcaption class="cmp-tag gen">Generated</figcaption></figure>';
  const [srcImg, genImg] = wrap.querySelectorAll('img');
  srcImg.src = srcUrl;
  genImg.src = genUrl;
}
