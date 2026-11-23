
document.addEventListener('DOMContentLoaded', () => {

  const loginForm = document.getElementById('loginForm');
  const registerForm = document.getElementById('registerForm');
  const forgotForm = document.getElementById('forgotForm');

  const title = document.getElementById('formTitle');
  const subtitle = document.getElementById('formSubtitle');
  const message = document.getElementById('message');

  // Display messages
  function showMessage(text, isError = true) {
    message.textContent = text;
    message.style.color = isError ? '#ff9999' : '#70efa5';
  }

  // Switch between forms
  function showForm(formToShow, heading, description) {
    loginForm.classList.add('hidden');
    registerForm.classList.add('hidden');
    forgotForm.classList.add('hidden');

    formToShow.classList.remove('hidden');

    title.textContent = heading;
    subtitle.textContent = description;

    showMessage('');
  }

  // LOGIN PAGE
  document.getElementById('showLogin').addEventListener('click', () => {
    showForm(
      loginForm,
      'Farmer Login',
      'Login to access the plant purity prediction system.'
    );
  });

  // REGISTER PAGE
  document.getElementById('showRegister').addEventListener('click', () => {
    showForm(
      registerForm,
      'Create Farmer Account',
      'Register to access the plant purity prediction system.'
    );
  });

  // FORGOT PASSWORD PAGE
  document.getElementById('showForgot').addEventListener('click', () => {
    showForm(
      forgotForm,
      'Reset Password',
      'Enter your registered email to request a password reset.'
    );
  });

  // BACK TO LOGIN
  document.getElementById('backToLogin').addEventListener('click', () => {
    showForm(
      loginForm,
      'Farmer Login',
      'Login to access the plant purity prediction system.'
    );
  });

  // SHOW / HIDE PASSWORD
  document.querySelectorAll('.toggle-password').forEach(button => {

    button.addEventListener('click', () => {

      const input = document.getElementById(button.dataset.target);

      if (!input) return;

      if (input.type === 'password') {

        input.type = 'text';

        button.setAttribute('aria-label', 'Hide password');
        button.setAttribute('title', 'Hide password');

        button.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 3l18 18"/>
            <path d="M10.6 10.6a2 2 0 002.8 2.8"/>
            <path d="M9.9 5.2A10.8 10.8 0 0112 5
              c6.5 0 10 7 10 7a14 14 0 01-3.2 4.1"/>
            <path d="M6.2 6.2C3.5 8.1 2 12 2 12s3.5 7
              10 7c1.5 0 2.8-.3 4-.8"/>
          </svg>
        `;

      } else {

        input.type = 'password';

        button.setAttribute('aria-label', 'Show password');
        button.setAttribute('title', 'Show password');

        button.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 12s3.5-7 10-7 10 7 10 7
              -3.5 7-10 7-10-7-10-7z"/>
            <circle cx="12" cy="12" r="3"/>
          </svg>
        `;
      }

    });

  });

  // SAFE SERVER RESPONSE HANDLER
  async function getResponseData(response) {

    const contentType = response.headers.get('content-type') || '';

    if (!contentType.includes('application/json')) {

      const responseText = await response.text();

      console.error('Unexpected server response:', responseText);

      if (response.status === 404) {
        throw new Error('Requested API route was not found. Please check the Flask backend route.');
      }

      if (response.status >= 500) {
        throw new Error('Server error occurred. Please check your Flask terminal.');
      }

      throw new Error('Server returned HTML instead of JSON. Check the backend route.');
    }

    let data;

    try {
      data = await response.json();
    } catch (error) {
      throw new Error('Invalid JSON response received from server.');
    }

    if (!response.ok) {
      throw new Error(data.error || 'Unable to continue.');
    }

    return data;
  }

  // SEND LOGIN / REGISTER REQUESTS
  async function sendAuth(url, payload) {

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    const data = await getResponseData(response);

    if (data.redirect) {
      window.location.href = data.redirect;
    } else {
      window.location.href = '/';
    }
  }

  // LOGIN
  loginForm.addEventListener('submit', async (event) => {

    event.preventDefault();
    showMessage('');

    try {

      await sendAuth('/auth/login', {
        email: document.getElementById('loginEmail').value.trim(),
        password: document.getElementById('loginPassword').value
      });

    } catch (error) {
      showMessage(error.message);
    }

  });

  // REGISTER
  registerForm.addEventListener('submit', async (event) => {

    event.preventDefault();
    showMessage('');

    try {

      await sendAuth('/auth/register', {
        name: document.getElementById('farmerName').value.trim(),
        email: document.getElementById('registerEmail').value.trim(),
        password: document.getElementById('registerPassword').value
      });

    } catch (error) {
      showMessage(error.message);
    }

  });

  // FORGOT PASSWORD
  forgotForm.addEventListener('submit', async (event) => {

    event.preventDefault();
    showMessage('');

    const email = document.getElementById('forgotEmail').value.trim();

    if (!email) {
      showMessage('Please enter your registered email address.');
      return;
    }

    try {

      const response = await fetch('/auth/forgot-password', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email: email })
      });

      const data = await getResponseData(response);

showMessage(data.message || 'Reset link generated.', false);

if (data.reset_url) {
    const link = document.createElement("a");

    link.href = data.reset_url;
    link.textContent = "Open Password Reset Link";
    link.target = "_blank";
    link.rel = "noopener noreferrer";

    link.style.display = "block";
    link.style.marginTop = "12px";
    link.style.color = "#70efa5";
    link.style.textDecoration = "underline";

    message.appendChild(link);
}

    } catch (error) {

      showMessage(
        error.message === 'Failed to fetch'
          ? 'Could not connect to the server. Make sure Flask is running.'
          : error.message
      );

    }

  });

});
