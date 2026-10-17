"""Lokales Vision-Language-Model über Ollama (Standard: qwen3-vl:4b-instruct).

Bewusst ohne Ollamas `format`-JSON-Schema: Die Grammatik-Beschränkung läuft
auf der CPU und war neben Gazebo 10–100× langsamer (gemessen auf einem M2
mit 16 GB). Stattdessen fordern die Prompts kurze JSON-Antworten an, die hier
tolerant geparst werden. Kurze Antworten sind entscheidend: ~15 Token
dauern 2–5 s, 250 Token über eine Minute.

Qwen3-VL gibt Bounding-Boxen als [x1, y1, x2, y2] relativ zur Bildgröße
im Bereich 0–1000 aus.

Teuer ist jedes neue Bild: Ollama skaliert es auf ~1100 Token, das dauert
neben Gazebo ~7–8 s (unabhängig von der Auflösung). Rückfragen zum selben
Bild als Fortsetzung des Chats (`history`) nutzen den Prompt-Cache (~1 s);
für Kontrollfragen taugt das beim 4B-Modell aber nicht – es bestätigt dann
meist nur seine erste Antwort.
"""
import base64
import json
import math
import os
import re
import time
import urllib.error
import urllib.request

import cv2

DEFAULT_MODEL = os.environ.get('VLM_MODEL', 'qwen3-vl:4b-instruct')
DEFAULT_HOST = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
if not DEFAULT_HOST.startswith('http'):
    DEFAULT_HOST = 'http://' + DEFAULT_HOST


def encode_image(img, max_side=640, quality=85):
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def parse_json(text):
    """Erstes JSON-Objekt/-Array aus einer Modellantwort (auch in ```json-Blöcken)."""
    text = re.sub(r'```(?:json)?', '', text)
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in '{[':
            try:
                return dec.raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                continue
    return None


class VLM:
    def __init__(self, model=DEFAULT_MODEL, host=DEFAULT_HOST, on_call=None, timeout=180):
        self.model = model
        self.host = host.rstrip('/')
        self.on_call = on_call  # Callback(dict) für Logging / Episodenaufzeichnung
        self.timeout = timeout
        self.calls = 0
        self.seconds = 0.0
        self.last_logprobs = None

    def check(self):
        """Läuft Ollama und ist das Modell da? Liefert eine Fehlermeldung oder None."""
        try:
            with urllib.request.urlopen(f'{self.host}/api/tags', timeout=5) as r:
                names = [m['name'] for m in json.load(r)['models']]
        except (urllib.error.URLError, OSError) as e:
            return f'Ollama nicht erreichbar unter {self.host} ({e}). Starten mit: ollama serve'
        if self.model not in names and f'{self.model}:latest' not in names:
            return f'Modell {self.model} fehlt. Laden mit: ollama pull {self.model}'
        return None

    def warmup(self):
        """Modell in den Speicher laden (der erste Aufruf dauert sonst ~30 s länger)."""
        self.chat('Answer with OK.', max_tokens=2, log=False)

    def chat(self, prompt, images=(), max_tokens=96, log=True, purpose='', history=None,
             logprobs=False):
        """Eine Nutzer-Nachricht senden. Nach dem Aufruf enthält `history` (falls
        übergeben) Frage und Antwort – für Rückfragen zum selben Bild.
        Mit logprobs=True steht die Antwort-Wahrscheinlichkeit des ersten Tokens
        in self.last_logprobs."""
        messages = history if history is not None else []
        messages.append({'role': 'user', 'content': prompt,
                         'images': [encode_image(i) for i in images]})
        body = {
            'model': self.model,
            'stream': False,
            'keep_alive': '60m',
            'options': {'temperature': 0, 'seed': 0, 'num_ctx': 4096, 'num_predict': max_tokens},
            'messages': messages,
        }
        if logprobs:
            body.update(logprobs=True, top_logprobs=10)
        t0 = time.monotonic()
        req = urllib.request.Request(f'{self.host}/api/chat', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            res = json.load(r)
        dt = time.monotonic() - t0
        text = res['message']['content'].strip()
        messages.append({'role': 'assistant', 'content': text})
        self.last_logprobs = (res.get('logprobs') or [None])[0]
        self.calls += 1
        self.seconds += dt
        if log and self.on_call:
            info = {'purpose': purpose, 'prompt': prompt, 'response': text,
                    'images': len(images), 'seconds': round(dt, 2),
                    'prompt_tokens': res.get('prompt_eval_count'),
                    'output_tokens': res.get('eval_count'), 'model': self.model}
            if logprobs and self.p_yes() is not None:
                info['p_yes'] = round(self.p_yes(), 4)
            self.on_call(info)
        return text

    def ask_json(self, prompt, images=(), max_tokens=96, purpose='', history=None):
        text = self.chat(prompt, images, max_tokens, purpose=purpose, history=history)
        return parse_json(text), text

    def ask_yes_no(self, prompt, images=(), purpose='', history=None, threshold=0.5):
        """(Antwort, P(yes)). Antwort ist True, wenn P(yes) >= threshold – mit
        threshold < 0.5 werden knappe Fälle zugunsten von "yes" entschieden."""
        text = self.chat(prompt + '\nAnswer only yes or no.', images, max_tokens=2,
                         purpose=purpose, history=history, logprobs=True).lower()
        p_yes = self.p_yes()
        if p_yes is None:  # Ollama ohne logprobs: nur der Text zählt
            p_yes = 1.0 if text.startswith('yes') else 0.0 if text.startswith('no') else None
        return (None if p_yes is None else p_yes >= threshold), p_yes

    def p_yes(self):
        """P(yes) unter den Kandidaten yes/no des ersten Antwort-Tokens."""
        lp = self.last_logprobs
        if not lp or not lp.get('top_logprobs'):
            return None
        yes = no = 0.0
        for cand in lp['top_logprobs']:
            tok = cand['token'].strip().lower()
            if tok == 'yes':
                yes += math.exp(cand['logprob'])
            elif tok == 'no':
                no += math.exp(cand['logprob'])
        return yes / (yes + no) if yes + no > 0 else None
