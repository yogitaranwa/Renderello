// Multi-file uploader: the dropzone is a native <label> bound to its hidden
// file input, so click, keyboard and drag-drop all work. Thumbnails carry
// per-file remove buttons and the selected count is announced.
export function makeUploader(dropId, inputId, thumbsId, max) {
  const drop = document.getElementById(dropId);
  const input = document.getElementById(inputId);
  const thumbs = document.getElementById(thumbsId);
  let files = [];

  const render = () => {
    thumbs.querySelectorAll('img').forEach(img => URL.revokeObjectURL(img.src));
    thumbs.innerHTML = '';
    files.forEach((f, i) => {
      const cell = document.createElement('span');
      cell.className = 'thumb';
      const img = document.createElement('img');
      img.src = URL.createObjectURL(f);
      img.alt = f.name;
      img.title = f.name;
      const rm = document.createElement('button');
      rm.type = 'button';
      rm.setAttribute('aria-label', 'Remove ' + f.name);
      rm.textContent = '✕';
      rm.onclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        files.splice(i, 1);
        render();
    input.dispatchEvent(new CustomEvent('fileschanged', { bubbles: true }));
      };
      cell.appendChild(img);
      cell.appendChild(rm);
      thumbs.appendChild(cell);
    });
    const title = drop.querySelector('.drop-title');
    if (title) {
      title.textContent = files.length
        ? files.length + ' of ' + max + ' selected'
        : title.dataset.default;
    }
  };

  const title = drop.querySelector('.drop-title');
  if (title) title.dataset.default = title.textContent;

  const add = (list) => {
    if (input.disabled) return;
    for (const f of list) {
      if (files.length >= max) break;
      if (!/^image\//.test(f.type)) continue;
      // Guard against double delivery of the same picker/drop event adding
      // one photo twice (duplicate files would generate duplicate variants).
      if (files.some(x => x.name === f.name && x.size === f.size &&
                          x.lastModified === f.lastModified)) continue;
      files.push(f);
    }
    render();
    input.dispatchEvent(new CustomEvent('fileschanged', { bubbles: true }));
  };

  drop.tabIndex = 0;
  drop.setAttribute('role', 'button');
  drop.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  input.onchange = () => { add(input.files); input.value = ''; };
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', e => {
    e.preventDefault();
    drop.classList.remove('over');
    add(e.dataTransfer.files);
  });

  return { get files() { return files; }, clear() { files = []; render(); input.dispatchEvent(new CustomEvent('fileschanged', { bubbles: true })); } };
}
