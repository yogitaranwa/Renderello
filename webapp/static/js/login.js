// Renderello Studio sign-in page. Self-contained: only login.js/login.css are
// served without a session, so this file must not import other modules.
//
// Contract with the backend:
//   GET  /api/login-mode  -> {mode: "password" | "code"}
//   POST /api/login       -> {ok: true} + session cookie
//     401 wrong credentials, 429 rate limited, 503 login not configured.
// Sessions are opaque server-side tokens invalidated on restart: a failed
// session means "sign in fresh", never an automatic credential re-post.
(async () => {
  const fields = document.getElementById('fields');
  const form = document.getElementById('login-form');
  const err = document.getElementById('err');
  const go = document.getElementById('go');
  const notice = document.getElementById('notice');

  // One-way context from the app shell (e.g. "session ended after restart").
  try {
    const flash = sessionStorage.getItem('renderello_flash');
    if (flash) {
      notice.textContent = flash;
      notice.hidden = false;
      sessionStorage.removeItem('renderello_flash');
    }
  } catch (e) { /* storage unavailable: skip the notice */ }

  let mode = 'password';
  try {
    const r = await fetch('/api/login-mode');
    if (!r.ok) throw new Error('status ' + r.status);
    const data = await r.json();
    mode = data.mode || 'password';
  } catch (e) {
    notice.textContent = 'Could not confirm the sign-in method (network or server error). The standard form is shown; if sign-in fails, wait a moment and try again.';
    notice.hidden = false;
  }

  let rememberedEmail = '';
  try { rememberedEmail = localStorage.getItem('renderello_email') || ''; } catch (e) { /* ok */ }

  const EYE_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
  const EYE_OFF_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/><path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/><path d="M14.12 14.12A3 3 0 1 1 9.88 9.88"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';

  if (mode === 'password') {
    fields.innerHTML =
      '<div class="field"><span><label for="email">Email</label></span>' +
      '<input id="email" type="email" autocomplete="username" required></div>' +
      '<div class="field"><span><label for="password">Password</label></span>' +
      '<div class="pw-wrap"><input id="password" type="password" autocomplete="current-password" required>' +
      '<button type="button" id="pw-toggle" class="pw-toggle" aria-pressed="false" aria-label="Show password">' + EYE_SVG + '</button></div></div>';
    if (rememberedEmail) {
      const em = document.getElementById('email');
      em.value = rememberedEmail; // non-secret convenience only; the password is never stored
    }
    const pwToggle = document.getElementById('pw-toggle');
    pwToggle.addEventListener('click', () => {
      const pw = document.getElementById('password');
      const show = pw.type === 'password';
      pw.type = show ? 'text' : 'password'; // value is preserved; default stays masked
      pwToggle.setAttribute('aria-pressed', String(show));
      pwToggle.setAttribute('aria-label', show ? 'Hide password' : 'Show password');
      pwToggle.innerHTML = show ? EYE_OFF_SVG : EYE_SVG;
      pw.focus();
    });
  } else {
    fields.innerHTML =
      '<div class="field"><span><label for="code">Access code</label></span>' +
      '<input id="code" type="password" autocomplete="off" required></div>';
  }

  form.onsubmit = async (e) => {
    e.preventDefault();
    err.textContent = '';
    go.disabled = true;
    go.classList.add('busy');
    const body = mode === 'password'
      ? { email: document.getElementById('email').value.trim(),
          password: document.getElementById('password').value }
      : { code: document.getElementById('code').value };
    try {
      const r = await fetch('/api/login', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      let data = null;
      try { data = await r.json(); } catch (e2) { /* handled below */ }
      if (r.ok) {
        if (mode === 'password') {
          try { localStorage.setItem('renderello_email', body.email); } catch (e3) { /* ok */ }
        }
        // Confirm the session cookie was actually stored before reloading;
        // some browsers (strict cookie blocking) drop it and would loop back
        // to this form with no visible error.
        let established = true;
        try {
          const probe = await fetch('/api/presets', { credentials: 'same-origin' });
          established = probe.ok;
        } catch (e4) { established = false; }
        if (established) {
          location.replace('/');
          return;
        }
        err.textContent = 'Signed in, but your browser did not keep the session cookie. Allow cookies for this site (Firefox: Settings, Privacy & Security, Cookies) or try a private window, then sign in again.';
        return;
      }
      if (data && data.detail) {
        err.textContent = data.detail; // wrong credentials / rate limit / not configured
      } else if (r.status === 401) {
        err.textContent = 'Wrong email or password.';
      } else if (r.status === 429) {
        err.textContent = 'Too many attempts. Wait 10 minutes and try again.';
      } else if (r.status === 503) {
        err.textContent = 'Sign-in is not configured on the server. Contact the administrator.';
      } else {
        err.textContent = 'Unexpected response from the server (' + r.status + '). Try again.';
      }
    } catch (e2) {
      err.textContent = 'Network error. The site may be waking up; try again in a few seconds.';
    } finally {
      go.disabled = false;
      go.classList.remove('busy');
    }
  };
})();
