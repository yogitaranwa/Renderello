// Results: run summary, honesty lines, and a comparison gallery. Every
// variant card can drag-compare the generated shot against the uploaded
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
    resultsBox.innerHTML = '';
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

  const srcUrl = sourceFiles.length ? URL.createObjectURL(sourceFiles[0]) : null;

  const grid = document.createElement('div');
  grid.className = 'variant-grid';
  host.appendChild(grid);

  for (const v of manifest.variants) {
    const card = document.createElement('div');
    card.className = 'vcard';
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

// Draggable source/generated comparison. The range input covers the frame so
// the divider follows pointer and keyboard alike; the after-image is clipped
// to the right of the divider.
function makeCompare(srcUrl, genUrl) {
  const wrap = document.createElement('div');
  wrap.className = 'compare';
  // A wipe slider only makes sense when both frames share an aspect; for
  // different aspects it letterboxes one image into a floating card and the
  // two products never line up. Those pairs render side by side instead:
  // both images fully visible, each in its own fitted panel.
  srcImg_probe(srcUrl, genUrl, (srcAspect, genAspect) => {
    if (Math.abs(srcAspect - genAspect) / Math.min(srcAspect, genAspect) > 0.03) {
      makePair(wrap, srcUrl, genUrl, Math.min(srcAspect, genAspect));
    }
  });
  wrap.innerHTML =
    '<img class="cmp-src" alt="Source product photo">' +
    '<div class="cmp-after"><img alt="Generated variant"></div>' +
    '<span class="cmp-tag src">Source</span>' +
    '<span class="cmp-tag gen">Generated</span>' +
    '<div class="cmp-handle"></div>' +
    '<input type="range" min="0" max="100" value="50" aria-label="Compare source and generated image">';
  const [srcImg, after, genImg, handle, range] = [
    wrap.querySelector('.cmp-src'), wrap.querySelector('.cmp-after'),
    wrap.querySelector('.cmp-after img'), wrap.querySelector('.cmp-handle'),
    wrap.querySelector('input'),
  ];
  srcImg.src = srcUrl;
  genImg.src = genUrl;
  // The frame matches the source photo's aspect so neither image is clipped
  // or distorted when source and generated aspects differ. Skipped entirely
  // if the aspect probe switched this card to side-by-side pair mode.
  const fitFrame = () => {
    if (wrap.classList.contains('pair')) return;
    if (srcImg.naturalWidth > 0 && srcImg.naturalHeight > 0) {
      wrap.style.aspectRatio = String(srcImg.naturalWidth) + ' / ' + String(srcImg.naturalHeight);
    }
  };
  if (srcImg.complete) fitFrame();
  srcImg.addEventListener('load', fitFrame);
  const apply = () => {
    if (wrap.classList.contains('pair')) return;
    const pct = Number(range.value);
    after.style.clipPath = 'inset(0 0 0 ' + pct + '%)';
    handle.style.left = pct + '%';
  };
  range.addEventListener('input', apply);
  apply();
  return wrap;
}

export function renderSkeletons(resultsBox, count) {
  resultsBox.innerHTML = '';
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
    resultsBox.innerHTML = '';
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

// Side-by-side fitted panels for source vs generated when their aspects
// differ: both fully visible, no wipe, no letterboxed overlay.
function makePair(wrap, srcUrl, genUrl, minAspect) {
  wrap.classList.add('pair');
  wrap.style.aspectRatio = String(2 * minAspect);
  wrap.innerHTML =
    '<figure class="pair-cell"><img alt="Source product photo"><figcaption class="cmp-tag src">Source</figcaption></figure>' +
    '<figure class="pair-cell"><img alt="Generated variant"><figcaption class="cmp-tag gen">Generated</figcaption></figure>';
  const [srcImg, genImg] = wrap.querySelectorAll('img');
  srcImg.src = srcUrl;
  genImg.src = genUrl;
}
