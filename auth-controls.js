document.addEventListener('DOMContentLoaded', () => {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = 'Logout';
  button.setAttribute('aria-label', 'Logout');
  button.style.cssText = 'position:fixed;top:18px;right:22px;z-index:9999;padding:9px 17px;border:1px solid #55d990;border-radius:8px;background:#07150f;color:#75f2a8;font:600 14px Outfit,sans-serif;cursor:pointer;';
  button.addEventListener('click', async () => {
    button.disabled = true;
    try { await fetch('/auth/logout', {method: 'POST'}); }
    finally { window.location.href = '/login'; }
  });
  document.body.appendChild(button);
});
