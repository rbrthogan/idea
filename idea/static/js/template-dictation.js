/* Free browser dictation. No app API key or audio upload endpoint is used. */
(function () {
  'use strict';

  function create({ input, button, status, preview }) {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const unavailable = !window.isSecureContext
      ? 'Dictation needs HTTPS or localhost. You can also use your device’s keyboard dictation.'
      : !Recognition ? 'Dictation is unavailable in this browser. Try Chrome or your device’s keyboard dictation.' : '';
    const idleMessage = 'Free dictation. Your browser may process speech online.';
    let active = null;
    let disabled = false;

    function render(message = idleMessage, error = false) {
      const stopping = !!active?.stopping;
      button.disabled = !!unavailable || disabled || stopping;
      button.setAttribute('aria-pressed', String(!!active));
      button.setAttribute('aria-label', active ? 'Stop dictation' : 'Dictate brief');
      button.title = unavailable || (active ? 'Stop dictation' : 'Dictate brief');
      button.dataset.state = active ? (stopping ? 'stopping' : 'listening') : 'idle';
      button.querySelector('i').className = active ? 'fas fa-stop' : 'fas fa-microphone';
      status.textContent = unavailable || message;
      status.dataset.state = error ? 'error' : active ? 'listening' : 'idle';
    }

    function clearPreview() {
      preview.textContent = '';
      preview.hidden = true;
    }

    function insert(text) {
      text = text.trim();
      if (!text || input.disabled) return;
      // Insert after any selection, preserving the user's existing words.
      const cursor = input.selectionEnd ?? input.value.length;
      const before = input.value.slice(0, cursor);
      const after = input.value.slice(cursor);
      const leading = before && !/\s$/.test(before) && !/^[,.;:!?)]/.test(text) ? ' ' : '';
      const trailing = after && !/^[\s,.;:!?)]/.test(after) ? ' ' : '';
      input.setRangeText(leading + text + trailing, cursor, cursor, 'end');
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }

    function finish(session, message, { keepInterim = false, error = false, canceled = false } = {}) {
      if (active !== session) return;
      active = null;
      clearTimeout(session.startTimer);
      clearTimeout(session.stopTimer);
      session.visibilityObserver?.disconnect();
      if (keepInterim) insert(session.interim);
      clearPreview();
      render(message, error);
      session.resolve(!canceled && !error);
    }

    function cancel() {
      if (!active) return;
      const session = active;
      finish(session, 'Dictation stopped.', { canceled: true });
      try { session.recognition.abort(); } catch (_) { /* Already ended. */ }
    }

    function stop() {
      if (!active) return Promise.resolve(true);
      const session = active;
      if (!session.stopping) {
        session.stopping = true;
        clearTimeout(session.startTimer);
        render('Finishing dictation…');
        // Some engines fail to deliver onend. Keep the last visible words and release the UI.
        session.stopTimer = setTimeout(() => {
          finish(session, 'Dictation stopped. Review your brief before generating.', { keepInterim: true });
          try { session.recognition.abort(); } catch (_) { /* Already ended. */ }
        }, 2500);
        try { session.recognition.stop(); }
        catch (_) { finish(session, 'Dictation stopped.', { keepInterim: true }); }
      }
      return session.done;
    }

    function start() {
      if (active || disabled || unavailable || input.disabled) return;
      let recognition;
      try { recognition = new Recognition(); }
      catch (_) { render('Dictation could not start. Try Chrome or your device’s keyboard dictation.', true); return; }
      const session = { recognition, interim: '', committed: new Set(), stopping: false, heard: false };
      session.done = new Promise(resolve => { session.resolve = resolve; });
      active = session;
      // App navigation can hide the compose panel without unloading the page.
      if (window.MutationObserver) {
        session.visibilityObserver = new window.MutationObserver(() => {
          if (active === session && (!input.getClientRects().length || input.disabled)) cancel();
        });
        for (let element = input; element; element = element.parentElement) {
          session.visibilityObserver.observe(element, { attributes: true, attributeFilter: ['style', 'class', 'hidden', 'disabled'] });
        }
      }
      recognition.lang = input.lang || window.navigator.language || document.documentElement.lang || 'en-US';
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;
      clearPreview();
      render('Starting microphone… Allow microphone access if prompted.');
      recognition.onstart = () => {
        if (active !== session || session.stopping) return;
        clearTimeout(session.startTimer);
        render('Listening… Click the stop button when you’re done.');
      };
      recognition.onresult = event => {
        if (active !== session) return;
        if (input.disabled) { cancel(); return; }
        const pending = [];
        for (let i = 0; i < event.results.length; i++) {
          const result = event.results[i];
          const text = result[0]?.transcript || '';
          if (result.isFinal) {
            if (!session.committed.has(i)) {
              session.committed.add(i);
              insert(text);
              session.heard ||= !!text.trim();
            }
          } else {
            pending.push(text);
          }
        }
        session.interim = pending.join(' ').trim();
        preview.textContent = session.interim;
        preview.hidden = !session.interim;
      };
      recognition.onerror = event => {
        if (active !== session) return;
        const messages = {
          'not-allowed': 'Microphone access was blocked. Allow microphone access in your browser settings, then try again.',
          'service-not-allowed': 'Speech recognition is blocked in this browser. Try Chrome or your device’s keyboard dictation.',
          'audio-capture': 'No microphone is available. Connect or enable a microphone, then try again.',
          'network': 'Dictation could not connect to the speech service. Check your connection and try again.',
          'no-speech': 'No speech was heard. Click the microphone and try again.',
          'language-not-supported': 'Your browser cannot dictate in its current language. Try your device’s keyboard dictation.',
          'aborted': 'Dictation stopped.'
        };
        finish(session, messages[event.error] || 'Dictation could not finish. Your brief is preserved; please try again.', { error: event.error !== 'aborted' });
        try { recognition.abort(); } catch (_) { /* Already ended. */ }
      };
      recognition.onend = () => finish(session,
        session.heard || session.interim ? 'Dictation stopped. Review your brief or click the microphone to continue.' : 'No speech was heard. Click the microphone and try again.',
        { keepInterim: true });
      session.startTimer = setTimeout(() => {
        finish(session, 'The microphone did not start. Check microphone permissions and try again.', { error: true });
        try { recognition.abort(); } catch (_) { /* Already ended. */ }
      }, 15000);
      try { recognition.start(); }
      catch (_) {
        finish(session, 'Dictation could not start. Check microphone permissions and try again.', { error: true });
        try { recognition.abort(); } catch (_) { /* Already ended. */ }
      }
    }

    button.addEventListener('click', () => { if (active) void stop(); else start(); });
    function escape(event) {
      if (event.key === 'Escape' && active) { event.preventDefault(); void stop(); }
    }
    button.addEventListener('keydown', escape);
    input.addEventListener('keydown', escape);
    document.addEventListener('visibilitychange', () => { if (document.hidden) cancel(); });
    window.addEventListener('pagehide', cancel);
    render();
    return {
      stop,
      cancel,
      setDisabled(value) {
        disabled = value;
        if (value) cancel();
        render();
      }
    };
  }

  window.IdeaDictation = { create };
})();
