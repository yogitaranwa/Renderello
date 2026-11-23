// Template gallery: native-button cards with generated cover art per template.
// Multi-select: up to MAX_TEMPLATES templates per batch; each card is a
// toggle button (aria-pressed), the selection count is announced in a live
// region, and per-template option values are kept per template so a batch
// sends each template its own knob_* values.
// Consumes GET /api/presets -> {presets: [{key,name,tagline,category,aspect,
// knobs:[{key,label,kind,options,default,hint}]}]} (contract unchanged).
const MAX_TEMPLATES = 4;

const COVERS = {
  clean_studio:       { h1: 222, h2: 265, icon: 'softbox' },
  luxury_hero:        { h1: 268, h2: 320, icon: 'diamond' },
  editorial_beauty:   { h1: 300, h2: 340, icon: 'sparkle' },
  lifestyle_kitchen:  { h1: 28,  h2: 48,  icon: 'mug' },
  natural_outdoor:    { h1: 140, h2: 96,  icon: 'leaf' },
  minimal_tabletop:   { h1: 210, h2: 180, icon: 'circle' },
  seasonal_holiday:   { h1: 350, h2: 300, icon: 'gift' },
  social_ad:          { h1: 200, h2: 250, icon: 'megaphone' },
  reflective_surface: { h1: 190, h2: 230, icon: 'mirror' },
  flat_lay:           { h1: 45,  h2: 20,  icon: 'grid' },
  macro_detail:       { h1: 260, h2: 210, icon: 'macro' },
  hero_negative_space:{ h1: 240, h2: 285, icon: 'frame' },
};
const FALLBACK_COVER = { h1: 250, h2: 290, icon: 'sparkle' };

const ICONS = {
  softbox: '<rect x="4" y="4" width="16" height="12" rx="2" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M12 16v4M8 21h8M8 8l8 4M16 8l-8 4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>',
  diamond: '<path d="M7 4h10l4 5-9 11L3 9l4-5z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M3 9h18M12 20L8.5 9l2-5M12 20l3.5-11-2-5" fill="none" stroke="currentColor" stroke-width="1.4"/>',
  sparkle: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3z" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/><path d="M18.5 15.5l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8.8-2.2z" fill="currentColor"/>',
  mug: '<path d="M5 8h11v8a4 4 0 01-4 4H9a4 4 0 01-4-4V8z" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M16 9.5h1.5a2.75 2.75 0 010 5.5H16" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M9 3.5c-.8 1 .8 1.6 0 2.6M13 3.5c-.8 1 .8 1.6 0 2.6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>',
  leaf: '<path d="M5 19C5 9 11 4 20 4c0 9-5 15-15 15z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M5 19c3-5 7-9 11-11" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>',
  circle: '<circle cx="12" cy="10" r="6" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M3 20h18" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
  gift: '<rect x="4" y="9" width="16" height="11" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M12 9v11M4 13h16" stroke="currentColor" stroke-width="1.6"/><path d="M12 9c-4 0-4.5-4-2-4.5S12.5 7 12 9zm0 0c4 0 4.5-4 2-4.5S11.5 7 12 9z" fill="none" stroke="currentColor" stroke-width="1.5"/>',
  megaphone: '<path d="M4 10v4l3 .5V9.5L4 10zm3-.5L17 6v12l-10-3.5v-5z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M19.5 9a4 4 0 010 6" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>',
  mirror: '<ellipse cx="12" cy="9" rx="6" ry="5" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M4 20c2-2 14-2 16 0" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path d="M8.5 15.5l7 2" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" opacity=".6"/>',
  grid: '<rect x="4" y="4" width="7" height="7" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.7"/><rect x="13" y="4" width="7" height="7" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.7"/><rect x="4" y="13" width="7" height="7" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.7"/><rect x="13" y="13" width="7" height="7" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.7"/>',
  macro: '<circle cx="10.5" cy="10.5" r="6" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M15.5 15.5L21 21" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="10.5" cy="10.5" r="2" fill="currentColor" opacity=".7"/>',
  frame: '<rect x="3.5" y="5" width="17" height="14" rx="2" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="9" cy="12" r="2.4" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M14 8.5h4M14 12h4" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" opacity=".7"/>',
};

function iconSvg(name) {
  return '<svg class="preset-icon" viewBox="0 0 24 24" aria-hidden="true">' + (ICONS[name] || ICONS.sparkle) + '</svg>';
}

export async function loadPresets(onChange, fetchApi) {
  const { presets } = await fetchApi('/api/presets');
  const grid = document.getElementById('presets');
  const knobsBox = document.getElementById('knobs');
  const countEl = document.getElementById('sel-count');
  const byKey = {};
  for (const p of presets) byKey[p.key] = p;

  const selected = [];              // preset keys, in click order
  const knobValues = {};            // key -> {knobKey: value}
  const cards = {};                 // key -> button element

  const defaultsFor = (p) => {
    const vals = {};
    for (const k of p.knobs.filter(k => k.key !== 'variants' && k.key !== 'aspect')) {
      vals[k.key] = k.kind === 'toggle' ? 'off' : (k.default || '');
    }
    return vals;
  };

  const updateCount = () => {
    if (!countEl) return;
    countEl.textContent = selected.length
      ? selected.length + (selected.length === 1 ? ' template' : ' templates') +
        ' selected (max ' + MAX_TEMPLATES + ')'
      : 'No templates selected yet.';
    countEl.classList.toggle('has-sel', selected.length > 0);
  };

  const renderKnobs = () => {
    knobsBox.innerHTML = '';
    if (!selected.length) {
      knobsBox.innerHTML = '<p class="hint">Choose one or more templates to see their options.</p>';
      return;
    }
    for (const key of selected) {
      const p = byKey[key];
      const group = document.createElement('div');
      group.className = 'knob-group';
      const h = document.createElement('h3');
      h.textContent = p.name;
      group.appendChild(h);
      const usable = p.knobs.filter(k => k.key !== 'variants' && k.key !== 'aspect');
      if (!usable.length) {
        const none = document.createElement('p');
        none.className = 'hint';
        none.textContent = 'No extra options.';
        group.appendChild(none);
        knobsBox.appendChild(group);
        continue;
      }
      for (const k of usable) {
        const div = document.createElement('div');
        div.className = 'knob';
        const id = 'knob-' + key + '-' + k.key;
        const label = document.createElement('label');
        label.setAttribute('for', id);
        label.textContent = k.label;
        div.appendChild(label);
        let field;
        if (k.kind === 'select') {
          field = document.createElement('select');
          for (const o of k.options) {
            const opt = document.createElement('option');
            opt.value = o;
            opt.textContent = o;
            if (o === knobValues[key][k.key]) opt.selected = true;
            field.appendChild(opt);
          }
        } else if (k.kind === 'toggle') {
          field = document.createElement('select');
          field.innerHTML = '<option>off</option><option>on</option>';
          field.value = knobValues[key][k.key] === 'on' ? 'on' : 'off';
        } else {
          field = document.createElement('input');
          field.type = 'text';
          field.value = knobValues[key][k.key] || '';
        }
        field.id = id;
        field.dataset.knob = key + ':' + k.key;
        field.addEventListener('input', () => { knobValues[key][k.key] = field.value; });
        field.addEventListener('change', () => { knobValues[key][k.key] = field.value; });
        div.appendChild(field);
        if (k.hint) {
          const hint = document.createElement('span');
          hint.className = 'knob-hint';
          hint.textContent = k.hint;
          div.appendChild(hint);
        }
        group.appendChild(div);
      }
      knobsBox.appendChild(group);
    }
  };

  const notify = (notice) => {
    updateCount();
    renderKnobs();
    onChange({ selected: [...selected], notice: notice || '' });
  };

  for (const p of presets) {
    const cover = COVERS[p.key] || FALLBACK_COVER;
    const el = document.createElement('button');
    el.type = 'button';
    el.className = 'preset';
    el.setAttribute('aria-pressed', 'false');
    el.innerHTML =
      '<span class="preset-cover" style="--h1:' + cover.h1 + ';--h2:' + cover.h2 + '">' +
        '<img class="preset-cover-img" src="/static/covers/' + encodeURIComponent(p.key) + '.jpg" alt="" loading="lazy">' +
        iconSvg(cover.icon) +
        '<span class="preset-check"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 13l4 4L19 7" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg></span>' +
        '<span class="preset-aspect"></span>' +
        '<span class="preset-preview">style preview</span>' +
      '</span>' +
      '<span class="preset-body">' +
        '<b></b>' +
        '<small></small>' +
        '<span class="preset-meta"></span>' +
      '</span>';
    el.querySelector('.preset-aspect').textContent = p.aspect || '';
    el.querySelector('.preset-meta').textContent = p.category || '';
    const coverImg = el.querySelector('.preset-cover-img');
    coverImg.addEventListener('load', () => el.querySelector('.preset-cover').classList.add('has-img'));
    coverImg.addEventListener('error', () => coverImg.remove());
    el.querySelector('b').textContent = p.name;
    el.querySelector('small').textContent = p.tagline;
    el.onclick = () => {
      const i = selected.indexOf(p.key);
      if (i >= 0) {
        selected.splice(i, 1);
        el.classList.remove('sel');
        el.setAttribute('aria-pressed', 'false');
        notify();
        return;
      }
      if (selected.length >= MAX_TEMPLATES) {
        updateCount();
        onChange({ selected: [...selected],
          notice: 'Up to ' + MAX_TEMPLATES + ' templates per batch. Deselect one to pick ' + p.name + '.' });
        return;
      }
      selected.push(p.key);
      knobValues[p.key] ||= defaultsFor(p);
      el.classList.add('sel');
      el.setAttribute('aria-pressed', 'true');
      notify();
    };
    grid.appendChild(el);
    cards[p.key] = el;
  }

  updateCount();
  renderKnobs();
  return {
    get selected() { return [...selected]; },
    knobValuesFor(key) { return Object.assign({}, knobValues[key] || {}); },
    nameFor(key) { return byKey[key] ? byKey[key].name : key; },
    deselect(key) { if (cards[key] && selected.includes(key)) cards[key].click(); },
    maxTemplates: MAX_TEMPLATES,
  };
}
