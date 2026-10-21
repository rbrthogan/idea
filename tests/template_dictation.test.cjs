const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../idea/static/js/template-dictation.js'), 'utf8');

function setup({ supported = true, secure = true, prefixed = false, throws = false } = {}) {
  function element() {
    return {
      value: '', selectionEnd: 0, textContent: '', hidden: false, disabled: false,
      attrs: {}, dataset: {}, events: {}, icon: {},
      addEventListener(name, fn) { this.events[name] = fn; },
      setAttribute(name, value) { this.attrs[name] = value; },
      querySelector() { return this.icon; },
      dispatchEvent(event) { this.events[event.type]?.(event); },
      setRangeText(text, start, end) {
        this.value = this.value.slice(0, start) + text + this.value.slice(end);
        this.selectionEnd = start + text.length;
      },
    };
  }
  const ui = { input: element(), button: element(), status: element(), preview: element() };
  const engines = [], timers = new Map(); let timerId = 0;
  class Recognition {
    constructor() { engines.push(this); }
    start() { if (throws) throw new Error('failed'); this.onstart(); }
    stop() { this.stopped = true; }
    abort() { this.aborted = true; this.onend?.(); }
    results(...items) {
      this.onresult({ results: items.map(([text, final]) => Object.assign([{ transcript: text }], { isFinal: final })) });
    }
  }
  const window = { ...element(), isSecureContext: secure, navigator: { language: 'en-IE' } };
  const document = { ...element(), documentElement: { lang: 'en' } };
  if (supported) window[prefixed ? 'webkitSpeechRecognition' : 'SpeechRecognition'] = Recognition;
  vm.runInNewContext(source, { window, document, Event, setTimeout(fn, ms) { timers.set(++timerId, { fn, ms }); return timerId; }, clearTimeout(id) { timers.delete(id); } });
  const controller = window.IdeaDictation.create(ui);
  return { ...ui, controller, engines, timers, document, window, click() { ui.button.events.click(); } };
}

test('supports the prefixed API and inserts once without losing selected or typed text', async () => {
  const s = setup({ prefixed: true });
  s.input.value = 'Existing tail'; s.input.selectionEnd = 8;
  let changes = 0; s.input.events.input = () => changes++;
  s.click(); const engine = s.engines[0];
  assert.equal(engine.lang, 'en-IE');
  assert.equal(engine.continuous, true);
  assert.equal(s.button.attrs['aria-pressed'], 'true');
  engine.results(['new idea', false]);
  assert.equal(s.input.value, 'Existing tail');
  assert.equal(s.preview.textContent, 'new idea');
  engine.results(['new idea', true]);
  engine.results(['new idea', true], ['more', false]);
  assert.equal(s.input.value, 'Existing new idea tail');
  assert.equal(changes, 1);
  const stopped = s.controller.stop();
  engine.results(['new idea', true], ['more', true]); engine.onend();
  await stopped;
  assert.equal(s.input.value, 'Existing new idea more tail');
  assert.equal(s.preview.hidden, true);
  assert.equal(s.button.attrs['aria-pressed'], 'false');
  assert.equal(s.timers.size, 0);
});

test('typing during dictation remains intact and the final result follows the current caret', () => {
  const s = setup(); s.click();
  s.input.value = 'Manually edited'; s.input.selectionEnd = s.input.value.length;
  s.engines[0].results(['spoken words', true]); s.engines[0].onend();
  assert.equal(s.input.value, 'Manually edited spoken words');
});

test('stop waits for final speech before a caller can submit the brief', async () => {
  const s = setup(); s.click(); const engine = s.engines[0];
  engine.results(['preliminary', false]);
  let submitted;
  const request = s.controller.stop().then(() => { submitted = s.input.value; });
  await Promise.resolve(); assert.equal(submitted, undefined);
  engine.results(['Final sentence.', true]); engine.onend(); await request;
  assert.equal(submitted, 'Final sentence.');
});

test('cancellation ignores late callbacks from an old session and clears interim text', () => {
  const s = setup(); s.click(); const old = s.engines[0]; old.results(['discard me', false]);
  s.controller.cancel(); s.click(); const next = s.engines[1];
  old.results(['wrong draft', true]); old.onerror({ error: 'network' }); old.onend();
  assert.equal(s.input.value, ''); assert.equal(s.button.attrs['aria-pressed'], 'true');
  next.results(['right draft', true]); next.onend();
  assert.equal(s.input.value, 'right draft'); assert.equal(old.aborted, true);
});

test('denied permission preserves the brief and allows retry', () => {
  const s = setup(); s.input.value = 'Keep this'; s.click();
  s.engines[0].onerror({ error: 'not-allowed' });
  assert.match(s.status.textContent, /Microphone access was blocked/);
  assert.equal(s.input.value, 'Keep this'); assert.equal(s.button.disabled, false);
  assert.equal(s.button.attrs['aria-pressed'], 'false'); assert.equal(s.timers.size, 0);
  s.click(); assert.equal(s.engines.length, 2); s.controller.cancel();
});

test('canceling while generation waits for dictation prevents submission', async () => {
  const s = setup(); s.click();
  const pending = s.controller.stop();
  s.controller.cancel();
  assert.equal(await pending, false);
});

test('unsupported and insecure browsers explain the disabled microphone', () => {
  for (const options of [{ supported: false }, { secure: false }]) {
    const s = setup(options); s.click();
    assert.equal(s.button.disabled, true); assert.equal(s.engines.length, 0);
    assert.match(s.status.textContent, options.secure === false ? /HTTPS or localhost/ : /unavailable in this browser/);
  }
});

test('hiding the page or locking the form aborts recording and prevents late insertion', () => {
  const s = setup(); s.click(); const first = s.engines[0];
  s.document.hidden = true; s.document.events.visibilitychange();
  assert.equal(first.aborted, true);
  s.click(); const second = s.engines[1]; s.controller.setDisabled(true);
  second.results(['late', true]);
  assert.equal(second.aborted, true); assert.equal(s.button.disabled, true);
  assert.equal(s.input.value, ''); s.controller.setDisabled(false);
  assert.equal(s.button.disabled, false);
});

test('a stalled stop preserves the visible interim text and releases the form', async () => {
  const s = setup(); s.click(); const engine = s.engines[0]; engine.results(['Last words', false]);
  const stopped = s.controller.stop();
  const timer = [...s.timers.values()].find(timer => timer.ms === 2500); timer.fn(); await stopped;
  assert.equal(s.input.value, 'Last words'); assert.equal(engine.aborted, true);
  assert.equal(s.button.disabled, false); assert.equal(s.timers.size, 0);
  engine.results(['Last words', true]); assert.equal(s.input.value, 'Last words');
});

test('a start failure resets the UI and Escape stops dictation', async () => {
  const failed = setup({ throws: true }); failed.click();
  assert.equal(failed.button.attrs['aria-pressed'], 'false'); assert.equal(failed.timers.size, 0);
  assert.match(failed.status.textContent, /could not start/);
  const s = setup(); s.click(); let prevented = false;
  s.input.events.keydown({ key: 'Escape', preventDefault() { prevented = true; } });
  assert.equal(prevented, true); assert.equal(s.engines[0].stopped, true);
  s.engines[0].onend(); assert.equal(s.button.attrs['aria-pressed'], 'false');
});
