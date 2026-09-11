"""Baixa boletins oficiais NEA; preserva HTML, URL e hash de cada fonte."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urljoin
import json
import re
import time
import subprocess

BASE = Path(__file__).resolve().parent / 'dados/energia_china'
RAW = BASE / 'fontes_html'
RAW.mkdir(parents=True, exist_ok=True)

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []; self.href = None; self.text = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a': self.href = dict(attrs).get('href'); self.text = []
    def handle_data(self, data):
        if self.href: self.text.append(data)
    def handle_endtag(self, tag):
        if tag == 'a' and self.href:
            self.links.append((self.href, ''.join(self.text).strip())); self.href = None

def baixar(url):
    url = url.replace('http://', 'https://')
    path = RAW / (sha256(url.encode()).hexdigest()[:20] + '.html')
    if not path.exists():
        for tentativa in range(3):
            try:
                body = subprocess.check_output(['curl', '--fail', '--location', '--silent', '--show-error', '--max-time', '40', url])
                path.write_bytes(body)
                break
            except Exception:
                if tentativa == 2: raise
                time.sleep(1)
    body = path.read_bytes()
    return {'url': url, 'arquivo': str(path.relative_to(BASE)),
            'sha256': sha256(body).hexdigest(), 'html': body.decode('utf-8-sig')}

def main():
    indices = ['https://www.nea.gov.cn/sjzz/ghs/zjgx' + (f'_{i}' if i > 1 else '') + '.htm' for i in range(1, 11)]
    artigos = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for item in pool.map(baixar, indices):
            p = Links(); p.feed(item['html'])
            for href, titulo in p.links:
                url = urljoin(item['url'], href).replace('http://', 'https://')
                match = re.search(r'nea.gov.cn/(20\d{2})', url)
                if '全社会用电量' in titulo and match and 2015 <= int(match[1]) <= 2025:
                    artigos[url] = titulo
    extra = BASE / 'fontes_adicionais.json'
    if extra.exists(): artigos.update(json.loads(extra.read_text()))
    print(f'{len(artigos)} boletins localizados.', flush=True)
    manifesto = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for item in pool.map(baixar, artigos):
            item.pop('html')
            item['titulo'] = artigos[item['url']]
            item['coleta_utc'] = datetime.now(timezone.utc).isoformat()
            manifesto.append(item)
    (BASE / 'manifesto_fontes.json').write_text(json.dumps(manifesto, ensure_ascii=False, indent=2))
    print(f'{len(manifesto)} boletins salvos em {BASE}.', flush=True)

if __name__ == '__main__': main()
