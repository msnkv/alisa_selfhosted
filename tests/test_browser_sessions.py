"""Выполняем JavaScript страницы с подменой DOM и HTTP (если есть Node.js)."""

import shutil
import subprocess
import unittest

from assistant.server import PAGE


@unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
class BrowserSessionTests(unittest.TestCase):
    def test_browser_identity_and_text_audio_headers(self):
        script = PAGE.split("<script>", 1)[1].split("</script>", 1)[0]
        harness = r"""
const vm = require('node:vm');
const assert = require('node:assert/strict');
const crypto = require('node:crypto').webcrypto;
let source = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => source += chunk);
process.stdin.on('end', async () => {
  function page(storage) {
    const requests = [];
    const element = () => ({append() {}, pause() {}, removeAttribute() {},
      addEventListener() {}, classList: {add() {}, remove() {}}});
    const elements = new Map();
    const context = vm.createContext({
      crypto, localStorage: storage, AbortController, TextDecoder,
      navigator: {}, window: {isSecureContext: false},
      document: {
        getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
        createElement: element, addEventListener() {},
      },
      location: {search: '?token=example'},
      fetch: async (url, options) => {
        requests.push({url, options});
        const data = new TextEncoder().encode(JSON.stringify({type:'done', text:'Ответ'}) + '\n');
        let sent = false;
        return {ok: true, body: {getReader: () => ({read: async () => {
          if (sent) return {done: true}; sent = true; return {value: data};
        }})}};
      },
    });
    vm.runInContext(source, context);
    return {context, requests, elements, id: vm.runInContext('sessionId', context)};
  }
  function storage() {
    const values = new Map();
    return {getItem: key => values.get(key) || null, setItem: (key, value) => values.set(key, value)};
  }
  const shared = storage();
  const first = page(shared);
  assert.match(first.id, /^browser:[a-f0-9]{32}$/);
  assert.equal(page(shared).id, first.id, 'после перезагрузки ID сохраняется');
  assert.notEqual(page(storage()).id, first.id, 'другой браузер получает другой ID');
  await vm.runInContext('send({method: "POST", headers: {"Content-Type": "application/json"}, body: "text"})', first.context);
  await vm.runInContext('send({method: "POST", headers: {"Content-Type": "audio/webm"}, body: "audio"})', first.context);
  assert.equal(first.requests.length, 2);
  for (const request of first.requests) {
    assert.equal(request.url, '/ask-stream?token=example');
    assert.equal(request.options.headers['X-Session-ID'], first.id);
    assert.match(request.options.headers['X-Request-ID'], /^[a-f0-9]{32}$/);
  }
  assert.equal(first.requests[0].options.headers['Content-Type'], 'application/json');
  assert.equal(first.requests[1].options.headers['Content-Type'], 'audio/webm');
  const broken = {getItem() {throw Error('disabled');}, setItem() {throw Error('disabled');}};
  assert.match(page(broken).id, /^browser:[a-f0-9]{32}$/, 'без localStorage страница работает');
  const invalid = {getItem() {return 'telegram:123';}, setItem() {}};
  assert.match(page(invalid).id, /^browser:[a-f0-9]{32}$/, 'неверный сохранённый ID заменяется');

  // Чтение NDJSON с произвольными границами байтов, порядок аудиофраз.
  const streaming = page(storage());
  const played = [];
  const player = streaming.elements.get('player');
  player.play = () => { played.push(player.src); queueMicrotask(() => player.onended()); return Promise.resolve(); };
  streaming.context.fetch = async () => {
    const data = new TextEncoder().encode([
      {type:'heard', text:'Вопрос'}, {type:'sentence', text:'Раз.', audio:'first'},
      {type:'sentence', text:'Два.', audio:'second'}, {type:'done'},
    ].map(e => JSON.stringify(e) + '\n').join(''));
    let position = 0;
    return {ok:true, body:{getReader:() => ({read:async () => {
      if (position >= data.length) return {done:true};
      const value = data.slice(position, position + 7); position += value.length; return {value};
    }})}};
  };
  await vm.runInContext('send({method:"POST",headers:{},body:"text"})', streaming.context);
  assert.deepEqual(played, ['data:audio/wav;base64,first', 'data:audio/wav;base64,second']);
  assert.equal(streaming.elements.get('stop').disabled, true);

  // Остановка посреди воспроизведения освобождает playPart и отменяет fetch.
  const stopping = page(storage());
  let started;
  const playback = new Promise(resolve => started = resolve);
  let pauses = 0;
  stopping.elements.get('player').play = () => {started(); return Promise.resolve();};
  stopping.elements.get('player').pause = () => pauses++;
  const requests = [];
  stopping.context.fetch = async (url, options) => {
    requests.push({url, options});
    if (url.startsWith('/cancel')) return {ok:true};
    let sent = false;
    const data = new TextEncoder().encode(JSON.stringify({type:'sentence', text:'Первая.', audio:'audio'}) + '\n');
    return {ok:true, body:{getReader:() => ({read:async () => sent ? {done:true} : (sent = true, {value:data})})}};
  };
  const sending = vm.runInContext('send({method:"POST",headers:{},body:"text"})', stopping.context);
  await playback;
  stopping.elements.get('stop').onclick();
  await sending;
  assert.equal(pauses, 1);
  assert.equal(requests.length, 2);
  assert.equal(requests[1].url, '/cancel?token=example');
  assert.equal(requests[1].options.headers['X-Request-ID'], requests[0].options.headers['X-Request-ID']);
  assert.equal(requests[0].options.signal.aborted, true);
  assert.equal(vm.runInContext('active', stopping.context), null);
  assert.equal(stopping.elements.get('q').disabled, false);
  console.log('Browser session JavaScript: OK');
});
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", harness], input=script,
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("JavaScript: OK", result.stdout)
