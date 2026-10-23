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
    const element = () => ({append() {}, addEventListener() {}, classList: {add() {}, remove() {}}});
    const elements = new Map();
    const context = vm.createContext({
      crypto, localStorage: storage,
      navigator: {}, window: {isSecureContext: false},
      document: {
        getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
        createElement: element, addEventListener() {},
      },
      location: {search: '?token=example'},
      fetch: async (url, options) => {
        requests.push({url, options});
        return {json: async () => ({text: 'Ответ'})};
      },
    });
    vm.runInContext(source, context);
    return {context, requests, id: vm.runInContext('sessionId', context)};
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
    assert.equal(request.url, '/ask?token=example');
    assert.equal(request.options.headers['X-Session-ID'], first.id);
  }
  assert.equal(first.requests[0].options.headers['Content-Type'], 'application/json');
  assert.equal(first.requests[1].options.headers['Content-Type'], 'audio/webm');
  const broken = {getItem() {throw Error('disabled');}, setItem() {throw Error('disabled');}};
  assert.match(page(broken).id, /^browser:[a-f0-9]{32}$/, 'без localStorage страница работает');
  const invalid = {getItem() {return 'telegram:123';}, setItem() {}};
  assert.match(page(invalid).id, /^browser:[a-f0-9]{32}$/, 'неверный сохранённый ID заменяется');
  console.log('Browser session JavaScript: OK');
});
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", harness], input=script,
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("JavaScript: OK", result.stdout)
