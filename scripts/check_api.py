"""Проверка API так, как это сделает жюри: загрузка видео, опрос статуса, проверка схемы ответа и времени.

Пример:
    python scripts/check_api.py --url https://xxxx.trycloudflare.com --key ВАШ_КЛЮЧ video1.mov video2.mov
Видео отправляются одновременно, как при массовой проверке. Лимит жюри — 5 минут на видео.
"""
import argparse
import base64
import sys
import threading
import time

import requests

LIMIT_SEC = 300


def check_schema(res):
    errors = []
    if res.get('verdict') not in ('violation', 'no_violation'):
        errors.append(f"verdict={res.get('verdict')!r}")
    viols = res.get('violations')
    if not isinstance(viols, list):
        return errors + ['violations не список']
    if res.get('verdict') == 'violation' and not viols:
        errors.append('verdict=violation, но список violations пуст')
    if res.get('verdict') == 'no_violation' and viols:
        errors.append('verdict=no_violation, но violations не пуст')
    for i, v in enumerate(viols):
        for field in ('type', 'timecode_sec', 'description', 'evidence_frames'):
            if field not in v:
                errors.append(f'violations[{i}] без поля {field}')
        frames = v.get('evidence_frames') or []
        if not 1 <= len(frames) <= 3:
            errors.append(f'violations[{i}]: кадров {len(frames)} (нужно 1-3)')
        for f in frames:
            try:
                if not base64.b64decode(f['image_base64'])[:3] == b'\xff\xd8\xff':
                    errors.append(f'violations[{i}]: кадр не JPEG')
            except Exception:
                errors.append(f'violations[{i}]: кадр не декодируется')
    return errors


def run_one(url, key, path, out):
    t0 = time.time()
    try:
        with open(path, 'rb') as fh:
            r = requests.post(f'{url}/api/v1/analyze', headers={'X-API-Key': key}, files={'file': (path.split('/')[-1].split('\\')[-1], fh, 'video/quicktime')}, timeout=120)
        r.raise_for_status()
        jid = r.json()['job_id']
        while time.time() - t0 < LIMIT_SEC + 60:
            time.sleep(5)
            res = requests.get(f'{url}/api/v1/jobs/{jid}', headers={'X-API-Key': key}, timeout=30).json()
            if res.get('status') != 'processing':
                break
        dt = time.time() - t0
        if res.get('status') != 'done':
            out[path] = f"❌ status={res.get('status')} {res.get('error', '')} ({dt:.0f} с)"
            return
        errs = check_schema(res)
        types = ', '.join(f"{v['type']} @ {v['timecode_sec']} с" for v in res['violations']) or '—'
        mark = '❌' if errs or dt > LIMIT_SEC else '✅'
        late = ' ПРЕВЫШЕН ЛИМИТ 5 МИН' if dt > LIMIT_SEC else ''
        out[path] = f"{mark} {res['verdict']} | {types} | {dt:.0f} с{late}" + (f" | ошибки: {'; '.join(errs)}" if errs else '')
    except Exception as exc:
        out[path] = f'❌ {exc}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True, help='публичный адрес сервиса, без / в конце')
    ap.add_argument('--key', required=True)
    ap.add_argument('videos', nargs='+')
    a = ap.parse_args()
    url = a.url.rstrip('/')
    bad = requests.post(f'{url}/api/v1/analyze', headers={'X-API-Key': 'wrong'}, files={'file': ('x.mov', b'0')}, timeout=30).status_code
    print(f"{'✅' if bad == 401 else '❌'} запрос с неверным ключом -> {bad} (должно быть 401)")
    out = {}
    threads = [threading.Thread(target=run_one, args=(url, a.key, v, out)) for v in a.videos]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for v in a.videos:
        print(f'{v}: {out.get(v)}')
    sys.exit(0 if all(str(x).startswith('✅') for x in out.values()) else 1)


if __name__ == '__main__':
    main()
