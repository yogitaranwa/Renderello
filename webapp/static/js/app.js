let matteReviewedFiles = [];
// Multi-template studio UI. Each selected template uses the single-preset
// generate API. Estimates are non-spending server reads, never local prices.
import { makeUploader } from './components/uploader.js';
import { loadPresets } from './components/presetGallery.js';
import { renderResults, renderSkeletons, renderError, clearResults } from './components/resultsGrid.js';

const $ = id => document.getElementById(id);
const products = makeUploader('drop-products', 'file-products', 'thumbs-products', 4);
const refs = makeUploader('drop-refs', 'file-refs', 'thumbs-refs', 4);
const status = $('status'), resultsBox = $('results');
const generateBtn = $('generate'), estimateEl = $('cost-estimate');
let gallery = null, busy = false, sessionLost = false, ready = false;
let estimateState = null, estimateVersion = 0;
let available = new Set();

for (const value of ['matte_plastic', 'glossy_plastic', 'glass', 'metal', 'fabric', 'ceramic', 'wood', 'leather']) {
  const option = document.createElement('option');
  option.value = value; option.textContent = value.replace('_', ' ');
  $('material').appendChild(option);
}

function setStatus(text, kind = '') {
  status.textContent = text || ''; status.className = kind;
}
function sessionEnded() {
  sessionLost = true;
  $('session-overlay').hidden = false;
  $('ov-signin').focus();
  $('ov-signin').onclick = () => {
    try { sessionStorage.setItem('renderello_flash', 'Your session ended or the server restarted. Please sign in again.'); } catch {}
    location.reload();
  };
  updateButton();
}
async function api(url, opts = {}) {
  let response;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);
  try { response = await fetch(url, { ...opts, signal: controller.signal }); }
  catch {
    throw new Error('Network error or timeout. The request may have reached the server. No automatic retry was sent.');
  } finally { clearTimeout(timeout); }
  if (response.status === 401) {
    sessionEnded();
    throw new Error('Session ended. Sign in again to continue.');
  }
  let data;
  try { data = await response.json(); }
  catch { throw new Error('Malformed response from the server. No automatic retry was sent.'); }
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : 'Request failed (' + response.status + ').';
    throw new Error(detail);
  }
  return data;
}

function selectionSpec(keys = gallery?.selected || []) {
  return {
    provider: $('provider').value,
    mh_model: $('mh_model').value || 'seedream-v5-flash',
    mh_route: $('mh_route').value || 'scene',
    knob_scene_max_attempts: $('scene_max_attempts').value || '1',
    variants: String(Math.max(1, Math.min(4, Number($('variants').value) || 4))),
    presets: keys.join(','),
  };
}
function specKey(spec) { return JSON.stringify(spec); }
function costKey(data) {
  return JSON.stringify([data.provider, data.model, data.images, data.credits_expected, data.credits_worst_case]);
}
function isPaid() { return $('provider').value === 'magichour'; }
const textFieldRules = [
  { id: 'instruction', name: 'Scene instruction' },
  { id: 'desc', name: 'Product description' },
];
const TEXT_LIMIT = 2000;
// Match Python's Unicode code-point count, not JavaScript UTF-16 length.
function characterCount(value) {
  let count = 0;
  for (const character of value) count++;
  return count;
}
function textBlock() {
  const invalid = textFieldRules.find(rule => characterCount($(rule.id).value) > TEXT_LIMIT);
  return invalid ? invalid.name + ' is over the 2,000-character limit. Shorten it before generating; your text has not been changed.' : '';
}
function installTextFeedback() {
  for (const rule of textFieldRules) {
    const field = $(rule.id), feedback = document.createElement('p');
    feedback.id = rule.id + '-feedback';
    field.setAttribute('aria-describedby', [field.getAttribute('aria-describedby'), feedback.id].filter(Boolean).join(' '));
    field.after(feedback);
    let previouslyInvalid = false;
    const refresh = () => {
      const wasInvalid = previouslyInvalid;
      const count = characterCount(field.value), over = count > TEXT_LIMIT;
      previouslyInvalid = over;
      field.setAttribute('aria-invalid', String(over));
      feedback.className = over ? 'notice-line fail' : 'hint';
      feedback.textContent = count.toLocaleString('en-US') + ' / 2,000 characters.' +
        (over ? ' Shorten this field before generating; your text has not been changed.' : '');
      updateButton();
      // A template/engine change while text was invalid may have cleared
      // its quote. Restore that missing quote once, not on every keystroke.
      if (wasInvalid && !textBlock() && !estimateState && gallery?.selected.length && !busy && !sessionLost) updateEstimate();
    };
    field.addEventListener('input', refresh);
    refresh();
  }
}
function localBlock() {
  const invalidText = textBlock();
  if (invalidText) return invalidText;
  if (isPaid() && products.files.length > 1) {
    return 'Magic Hour scene composition supports one product photo per run. Remove extra product photos or choose the procedural engine.';
  }
  if (isPaid() && $('mh_route').value === 'whole' && refs.files.length) {
    return 'Magic Hour whole-image generation accepts one input image only. Remove reference photos or choose the procedural engine. Multi-input pricing is not verified.';
  }
  return '';
}
function updateButton() {
  const spec = selectionSpec();
  const hasSelection = !!gallery?.selected.length;
  const quoted = estimateState && estimateState.key === specKey(spec) && estimateState.allowed;
  generateBtn.disabled = busy || sessionLost || !ready || !!localBlock() || (hasSelection && !quoted);
  generateBtn.classList.toggle('busy', busy);
  if (!busy) {
    const worst = estimateState?.data?.credits_worst_case;
    generateBtn.querySelector('.btn-label').textContent = isPaid() && quoted && worst != null
      ? 'Generate · up to ' + worst + ' credits' : 'Generate photoshoot';
  }
}
function displayEstimate(data) {
  if (data.free === true && data.credits === 0) {
    estimateEl.textContent = 'This batch makes ' + data.images + ' images on the ' + data.provider + ' engine: no credits.';
    return true;
  }
  const budget = data.budget || {};
  const numbers = [data.credits_per_image, data.credits_expected, data.credits_worst_case,
    data.images, budget.remaining, budget.max_batch];
  if (data.provider !== 'magichour' || !numbers.every(x => typeof x === 'number' && Number.isFinite(x)) || data.credits_per_image <= 0) {
    throw new Error('The server did not return a valid model price and budget. Paid generation is disabled.');
  }
  let text = data.images + (data.images === 1 ? ' image. Expected: ' : ' images. Expected: ') + data.credits_expected + ' credits; up to ' +
    data.credits_worst_case + ' with background retries. Budget remaining: ' + budget.remaining +
    ' credits; per-batch limit: ' + budget.max_batch + '.';
  if (budget.allocated != null && budget.spent != null) {
    text += ' Used ' + budget.spent + ' of ' + budget.allocated + '.';
  }
  const withinLimits = data.within_budget === true && data.within_batch_limit === true;
  const modelAllowed = data.model_allowed === true;
  const allowed = withinLimits && modelAllowed;
  if (!modelAllowed) text += ' This model is not enabled for live use on this deployment. Choose an enabled model.';
  if (!withinLimits) text += ' This selection exceeds the budget or batch limit. Reduce templates or variants.';
  text += ' Price preview only; provider balance and safety checks run again before spending.';
  estimateEl.textContent = text;
  return allowed;
}
async function updateEstimate() {
  const version = ++estimateVersion;
  estimateState = null;
  const keys = gallery?.selected || [];
  if (!keys.length) {
    estimateEl.textContent = 'Choose templates to preview this batch.';
    updateButton(); return null;
  }
  const block = localBlock();
  if (block) { estimateEl.textContent = block; updateButton(); return null; }
  const spec = selectionSpec(keys), key = specKey(spec);
  estimateEl.textContent = 'Checking batch cost and budget…';
  updateButton();
  try {
    const data = await api('/api/estimate?' + new URLSearchParams(spec));
    if (version !== estimateVersion) return null;
    if (data.provider !== spec.provider || (spec.provider === 'magichour' && data.free === true)) throw new Error('Estimate engine mismatch.');
    const allowed = displayEstimate(data);
    estimateState = { key, data, allowed };
    updateButton(); return estimateState;
  } catch (error) {
    if (version !== estimateVersion) return null;
    estimateEl.textContent = 'Cost check unavailable: ' + error.message;
    updateButton(); return null;
  }
}

function renderChips() {
  const host = $('chosen-chips');
  host.replaceChildren();
  for (const key of gallery?.selected || []) {
    const name = gallery.nameFor(key);
    const chip = document.createElement('span'); chip.className = 'chip';
    const text = document.createElement('span'); text.textContent = name;
    const remove = document.createElement('button');
    remove.type = 'button'; remove.textContent = '✕';
    remove.setAttribute('aria-label', 'Remove ' + name + ' from the batch');
    remove.disabled = busy;
    remove.onclick = () => {
      gallery.deselect(key);
      ($('chosen-chips').querySelector('button') || generateBtn).focus();
    };
    chip.append(text, remove); host.append(chip);
  }
}
function setBusy(value, label = 'Generating') {
  busy = value;
  document.querySelectorAll('main input, main textarea, main select, .preset, .chip button, #thumbs-products button, #thumbs-refs button').forEach(el => {
    if (value) { el.dataset.batchDisabled = String(el.disabled); el.disabled = true; }
    else if (el.dataset.batchDisabled != null) {
      el.disabled = el.dataset.batchDisabled === 'true'; delete el.dataset.batchDisabled;
    }
  });
  if (value) generateBtn.querySelector('.btn-label').textContent = label;
  updateButton();
}

async function boot() {
  try {
    const data = await api('/api/providers');
    if (!Array.isArray(data.providers)) throw new Error('Malformed engine list.');
    const select = $('provider'), models = $('mh_model');
    for (const provider of data.providers) {
      const option = document.createElement('option');
      option.value = provider.name;
      option.textContent = provider.name + (provider.available ? '' : ' (unavailable: ' + provider.unavailable_reason + ')');
      option.disabled = !provider.available;
      select.appendChild(option);
      if (provider.available) available.add(provider.name);
    }
    select.value = available.has('procedural') ? 'procedural' : ([...available][0] || '');
    const mh = data.providers.find(p => p.name === 'magichour');
    $('provider-badge').textContent = 'Engines ready: ' + ([...available].join(', ') || 'none') +
      (mh?.available ? ' · Magic Hour uses credits' : ' · Magic Hour unavailable');
    models.disabled = !mh?.available;
    if (mh?.available) {
      for (const [model, price] of Object.entries(mh.models || {})) {
        const option = document.createElement('option'); option.value = model;
        const enabled = Array.isArray(mh.live_models) && mh.live_models.includes(model);
        option.disabled = !enabled;
        option.textContent = model + ' (' + price + ' cr/image before retries)' + (enabled ? '' : ' · not enabled for live use');
        models.appendChild(option);
      }
      const firstEnabled = [...models.options].find(option => !option.disabled);
      models.value = firstEnabled?.value || '';
      models.disabled = !firstEnabled;
    } else {
      const option = document.createElement('option'); option.value = 'seedream-v5-flash';
      option.textContent = 'unavailable'; models.appendChild(option);
    }
    gallery = await loadPresets(({ notice }) => {
      setStatus(notice || '', notice ? 'err' : '');
      renderChips(); updateEstimate();
    }, api);
    ready = available.size > 0;
  } catch (error) {
    setStatus('Could not load the studio: ' + error.message, 'err');
  }
  for (const id of ['provider', 'variants', 'mh_model', 'mh_route', 'scene_max_attempts']) $(id).addEventListener('change', updateEstimate);
  for (const id of ['file-products', 'file-refs']) $(id).addEventListener('fileschanged', updateEstimate);
  installTextFeedback();
  generateBtn.onclick = submit;
  $('logout').onclick = async () => {
    if (busy) { setStatus('Wait for the active batch before signing out.', 'err'); return; }
    try { await api('/api/logout', { method: 'POST' }); location.reload(); }
    catch (error) { setStatus(error.message, 'err'); }
  };
  if (window.ResizeObserver) {
    new ResizeObserver(entries => {
      document.documentElement.style.setProperty('--dock-height', Math.ceil(entries[0].target.getBoundingClientRect().height + 24) + 'px');
    }).observe(document.querySelector('.dock'));
    new ResizeObserver(entries => {
      document.documentElement.style.setProperty('--header-height', Math.ceil(entries[0].target.getBoundingClientRect().height) + 'px');
    }).observe(document.querySelector('.topbar'));
  }
  updateButton();
}

function snapshotForm(key) {
  const form = new FormData();
  for (const file of products.files) form.append('products', file);
  for (const file of refs.files) form.append('references', file);
  form.append('preset', key);
  for (const [field, id] of Object.entries({ provider: 'provider', instruction: 'instruction',
    product_description: 'desc', material: 'material', aspect: 'aspect', variants: 'variants', mh_model: 'mh_model', mh_route: 'mh_route' })) {
    form.append(field, $(id).value);
  }
  if (matteReviewedFiles.length === products.files.length && matteReviewedFiles.every((f, i) => f === products.files[i])) form.append('knob_matte_reviewed', '1');
  form.append('knob_scene_max_attempts', $('scene_max_attempts').value || '1');
  for (const [name, value] of Object.entries(gallery.knobValuesFor(key))) form.append('knob_' + name, value);
  return form;
}
async function poll(jobId, label) {
  const started = Date.now();
  for (;;) {
    if (Date.now() - started > 30 * 60 * 1000) {
      throw new Error('Run status timed out. The server may still be working. No further templates were started.');
    }
    await new Promise(resolve => setTimeout(resolve, 1500));
    const job = await api('/api/jobs/' + encodeURIComponent(jobId));
    if (job.status === 'running' || job.status === 'queued') {
      setStatus(label + ' · ' + Math.round((Date.now() - started) / 1000) + 's elapsed', 'working');
      continue;
    }
    if (job.status === 'failed') { const error = new Error(job.error || 'Run failed.'); error.matteReview = job.matte_review; throw error; }
    if (job.status !== 'done') throw new Error('Unknown job status. No further templates were started.');
    const manifests = Array.isArray(job.runs) && job.runs.length
      ? job.runs.map(run => run.manifest || run) : [job.manifest];
    if (!manifests.every(m => m && Array.isArray(m.variants) && m.variants.length && m.run_dir &&
        m.variants.every(v => v && (v.error || (typeof v.image === 'string' && v.image))))) {
      throw new Error('Malformed run results. No automatic retry was sent.');
    }
    return manifests;
  }
}
async function submit() {
  if (busy || sessionLost) return;
  const keys = gallery?.selected || [];
  if (!keys.length) { setStatus('Choose at least one template first.', 'err'); $('template-heading').focus(); return; }
  if (!products.files.length) { setStatus('Add at least one product photo.', 'err'); $('drop-products').focus(); return; }
  if (localBlock()) { setStatus(localBlock(), 'err'); return; }
  if (!available.has($('provider').value)) { setStatus('Choose an available engine.', 'err'); return; }
  const priorQuote = estimateState;
  const sourceFiles = [...products.files];
  const forms = keys.map(snapshotForm);
  setBusy(true, 'Checking batch…');
  let completed = 0, rendered = false;
  try {
    const quote = await updateEstimate();
    if (!quote?.allowed) throw new Error('Batch cost could not be approved. Check the estimate above. Nothing was generated.');
    if (isPaid() && (!priorQuote || priorQuote.key !== quote.key || costKey(priorQuote.data) !== costKey(quote.data))) {
      setStatus('The batch cost changed. Review the updated expected and maximum cost, then click Generate again.', 'err');
      return;
    }
    renderSkeletons(resultsBox, Math.min(8, keys.length * Number($('variants').value)));
    for (let i = 0; i < keys.length; i++) {
      const name = gallery.nameFor(keys[i]);
      const label = 'Template ' + (i + 1) + ' of ' + keys.length + ': ' + name;
      generateBtn.querySelector('.btn-label').textContent = 'Generating ' + (i + 1) + ' of ' + keys.length;
      setStatus(label, 'working');
      const job = await api('/api/generate', { method: 'POST', body: forms[i] });
      if (typeof job.job_id !== 'string' || !job.job_id) throw new Error('The server did not return a job ID. No automatic retry was sent.');
      const manifests = await poll(job.job_id, label);
      if (!rendered) { clearResults(resultsBox); rendered = true; }
      for (const manifest of manifests) renderResults(manifest, resultsBox, sourceFiles, { append: true, title: name });
      const failed = manifests.flatMap(manifest => manifest.variants).filter(variant => variant.error);
      if (failed.length) {
        throw new Error(name + ': ' + failed.length + ' variant(s) failed. Any available results are kept below. No remaining templates were started.');
      }
      completed++;
    }
    setStatus('Batch done: ' + completed + ' of ' + keys.length + ' templates rendered. Eye review is still required.', 'ok');
  } catch (error) {
    // Never repeat an uncertain paid request, continue spending after a
    // failure, or erase completed groups when a later template fails.
    if (!rendered) clearResults(resultsBox);
    renderError(resultsBox, error.message, { append: true, title: 'Batch stopped' });
    if (error.matteReview) {
      const panel = document.createElement('section');
      const title = document.createElement('h3');
      title.textContent = 'Cutout review: source left, mask right';
      panel.append(title);
      for (const name of error.matteReview.previews) {
        const image = document.createElement('img');
        image.src = '/api/runs/' + encodeURIComponent(error.matteReview.run_dir) + '/' + encodeURIComponent(name);
        image.alt = 'Source and cutout mask. White is kept; black is removed.';
        image.style.maxWidth = '100%';
        panel.append(image);
      }
      const note = document.createElement('p');
      note.textContent = 'This check cannot prove all products are present. Missing items will not be restored. No generation was started for this template.';
      const accept = document.createElement('button');
      accept.type = 'button';
      accept.className = 'btn-primary';
      accept.style.whiteSpace = 'normal';
      accept.style.height = 'auto';
      accept.style.padding = '12px 18px';
      accept.textContent = 'I reviewed these masks and accept this cutout';
      accept.onclick = () => {
        if (sourceFiles.length !== products.files.length || !sourceFiles.every((f, i) => f === products.files[i])) {
          accept.textContent = 'Sources changed. Generate a new review.';
          return;
        }
        matteReviewedFiles = sourceFiles;
        accept.disabled = true;
        accept.textContent = 'Confirmed for these photos. Click Generate to continue with the current cost.';
      };
      panel.append(note, accept);
      resultsBox.append(panel);
    }
    setStatus('Batch stopped: ' + completed + ' of ' + keys.length + ' completed. ' + error.message, 'err');
  } finally {
    setBusy(false);
    if (!sessionLost) updateEstimate();
  }
}
boot();
