"""Extrai os valores explícitos dos boletins NEA; não imputa meses ausentes."""
from html.parser import HTMLParser
from pathlib import Path
import json
import re
import pandas as pd

BASE = Path(__file__).resolve().parent / 'dados/energia_china'

class Texto(HTMLParser):
    def __init__(self):
        super().__init__(); self.partes = []; self.ignorar = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'): self.ignorar += 1
        if tag in ('p', 'div', 'br', 'td', 'h1'): self.partes.append('\n')
    def handle_endtag(self, tag):
        if tag in ('script', 'style'): self.ignorar -= 1
        if tag in ('p', 'div', 'td', 'h1'): self.partes.append('\n')
    def handle_data(self, data):
        if not self.ignorar: self.partes.append(data)

def main():
    manifesto = json.loads((BASE / 'manifesto_fontes.json').read_text())
    registros = []
    for fonte in manifesto:
        p = Texto(); p.feed((BASE / fonte['arquivo']).read_text())
        texto = '\n'.join(re.sub(r'\s+', '', s) for s in ''.join(p.partes).splitlines())
        data = re.search(r'发布时间[：:]?(20\d{2}-\d{2}-\d{2})', texto)
        if not data: raise ValueError(f"Data não localizada: {fonte['url']}")
        publicacao = pd.Timestamp(data[1])
        ano_ref = publicacao.year - (publicacao.month == 1)
        padroes = {
            'mensal': r'(?:^|[。；])(?:(20\d{2})年)?(\d{1,2})月份?，(?:全国|我国[^。；\n]*?)?(?:全社会)?用电量[^。；\n]*?(\d+(?:\.\d+)?)亿千瓦时',
            'acumulado': r'(?:^|[。；])(?:(20\d{2})年|今年)?1[-~～—至－](\d{1,2})月份?，[^。；\n]*?全社会用电量[^。；\n]*?(\d+(?:\.\d+)?)亿千瓦时',
            'anual': r'(?:^|[。；])(?:(20\d{2})年|全年)，[^。；\n]*?全社会用电量[^。；\n]*?(\d+(?:\.\d+)?)亿千瓦时',
        }
        for tipo, padrao in padroes.items():
            for m in re.finditer(padrao, texto, re.M):
                ano = int(m[1]) if m[1] else ano_ref
                mes = 12 if tipo == 'anual' else int(m[2])
                valor = float(m[2] if tipo == 'anual' else m[3])
                if not (2015 <= ano <= 2024 and 1 <= mes <= 12): continue
                registros.append({'ano': ano, 'mes': mes, 'tipo': tipo,
                    'valor_100_milhoes_kwh': valor, 'valor_twh': valor / 10,
                    'publicacao': str(publicacao.date()), 'url': fonte['url'],
                    'arquivo_fonte': fonte['arquivo'], 'trecho_original': m[0].lstrip('。；')})
    r = pd.DataFrame(registros).drop_duplicates()
    r.to_csv(BASE / 'valores_extraidos.csv', index=False)
    mensal = r[r.tipo.eq('mensal')].copy()
    assert not mensal.duplicated(['ano', 'mes']).any(), 'Revisar múltiplas fontes do mesmo mês.'
    mensal['data'] = pd.to_datetime(dict(year=mensal.ano, month=mensal.mes, day=1))
    grade = pd.DataFrame({'data': pd.date_range('2015-01-01', '2024-12-01', freq='MS')})
    grade = grade.merge(mensal.drop(columns=['ano', 'mes', 'tipo']), on='data', how='left')
    grade['status'] = grade.valor_twh.notna().map({True: 'observado_mensal', False: 'nao_localizado_mensal'})
    grade['observado_direto_twh'] = grade.valor_twh
    grade['nota'] = ''
    grade.loc[grade.url.fillna('').str.contains('obor.nea.gov.cn', regex=False), 'nota'] = 'Número atribuído à NEA em republicação do People\u0027s Daily no portal de cooperação energética da NEA.'
    # Janeiro só é recuperado quando fevereiro e o bimestre constam da MESMA fonte.
    # Trata-se de identidade contábil sobre valores publicados, não imputação estatística.
    for ano in range(2015, 2025):
        fev = mensal[(mensal.ano == ano) & (mensal.mes == 2)]
        bim = r[(r.ano == ano) & (r.mes == 2) & r.tipo.eq('acumulado')]
        jan_idx = grade.index[grade.data.eq(pd.Timestamp(ano, 1, 1))][0]
        if len(fev) == len(bim) == 1 and fev.iloc[0].url == bim.iloc[0].url and pd.isna(grade.loc[jan_idx, 'valor_twh']):
            f, b = fev.iloc[0], bim.iloc[0]
            valor = b.valor_100_milhoes_kwh - f.valor_100_milhoes_kwh
            grade.loc[jan_idx, ['valor_100_milhoes_kwh', 'valor_twh']] = [valor, valor / 10]
            for col in ['publicacao', 'url', 'arquivo_fonte']: grade.loc[jan_idx, col] = b[col]
            grade.loc[jan_idx, 'trecho_original'] = b.trecho_original + ' | ' + f.trecho_original
            grade.loc[jan_idx, 'status'] = 'reconstituido_mesma_fonte'
            grade.loc[jan_idx, 'nota'] = f'Janeiro = bimestre ({b.valor_100_milhoes_kwh:g}) - fevereiro ({f.valor_100_milhoes_kwh:g}), em 10^8 kWh; sujeito ao arredondamento da fonte.'
    faltantes = grade.valor_twh.isna()
    grade.loc[faltantes, 'nota'] = 'Valor mensal não localizado nos boletins coletados; não significa consumo zero nem inexistência em outras bases.'
    grade.loc[faltantes & grade.data.dt.month.eq(12), 'nota'] += ' Há total anual separado; diferença entre publicações pode conter revisões.'
    grade.loc[grade.data.isin(pd.to_datetime(['2024-01-01','2024-02-01'])), 'nota'] = 'Boletim divulga apenas janeiro–fevereiro: 1531,6 TWh; divisão mensal não identificada nesta coleta.'
    grade.to_csv(BASE / 'consumo_eletricidade_china_mensal.csv', index=False)
    print('Valores extraídos por tipo:', r.tipo.value_counts().to_dict())
    print('Meses ausentes:', grade.loc[grade.valor_twh.isna(), 'data'].dt.strftime('%Y-%m').tolist())
    print(grade.groupby(grade.data.dt.year).valor_twh.agg(['count', 'min', 'max']).to_string())

if __name__ == '__main__': main()
