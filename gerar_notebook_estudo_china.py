"""Gera e executa o notebook didático; usa apenas dados locais da coleta NEA."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-estudo-china')
import base64
import contextlib
import io
import json
import sys
import uuid
from pathlib import Path

CELULAS = []
def md(s):
    CELULAS.append(dict(cell_type='markdown', id=uuid.uuid4().hex[:8], metadata={}, source=s.strip().splitlines(True)))
def code(s):
    CELULAS.append(dict(cell_type='code', id=uuid.uuid4().hex[:8], metadata={}, source=s.strip().splitlines(True), execution_count=None, outputs=[]))

md(r'''
# Consumo de eletricidade na China — estudo completo de séries temporais
**Econometria II · 2015–2024 · análise preparada em 11/09/2026**

Este notebook explica e executa: estatística descritiva, remoção e reposição de observações, testes de estacionariedade, identificação e tratamento de outliers, médias móveis, decomposição em tendência/sazonalidade/ciclo e suavização de Holt e Holt-Winters.

**Ideia central:** uma série temporal é uma sequência de valores em ordem de tempo. Aqui, cada valor é a eletricidade consumida na China em um mês. A ordem importa: agosto de 2020 pode se parecer tanto com julho de 2020 quanto com agosto de outros anos.

## Roteiro
1. Dados e estatística descritiva.
2. Experimento de imputação: apagar valores conhecidos, repor e medir o erro.
3. Preparação de uma série de trabalho completa.
4. Testes ADF e KPSS de estacionariedade.
5. Outliers reais suspeitos e experimento com erros artificiais.
6. Média móvel e decomposição: tendência, sazonalidade, ciclo e resíduo.
7. Holt e Holt-Winters: funcionamento, ajuste e comparação fora da amostra.
8. Conclusões automáticas e arquivos de resultados.

A ordem começa pelo preenchimento porque vários métodos exigem uma série sem lacunas. **Isso não transforma estimativas em observações reais.** As máscaras de procedência são mantidas em todas as etapas.

### Como executar
Mantenha este notebook junto à pasta `dados/energia_china`. Use Python com `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib` e `IPython`; execute as células na ordem. Não é necessário acesso à internet: os dados da coleta anterior estão salvos. As tabelas e figuras já estão incorporadas ao notebook. Para refazer a coleta original, consulte `coletar_energia_china.py` e o notebook descritivo anterior.
''')
code(r'''
from pathlib import Path
import json
import hashlib
import inspect
import warnings
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import statsmodels
from scipy.interpolate import PchipInterpolator
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing
from IPython.display import display, Markdown

RAIZ = Path.cwd()
BASE = RAIZ / 'dados/energia_china'
SAIDA = RAIZ / 'resultados/energia_china'
FIG = SAIDA / 'figuras'
FIG.mkdir(parents=True, exist_ok=True)
SEMENTE = 20260911
plt.rcParams.update({'figure.figsize': (12, 4.5), 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 150)

def tabela(df, nome, casas=3):
    df.to_csv(SAIDA / (nome + '.csv'), index=True)
    display(df.round(casas))

def figura(nome):
    plt.gcf().tight_layout()
    plt.gcf().savefig(FIG / (nome + '.png'), dpi=150, bbox_inches='tight')
    plt.show()

print('Versões:', {'Python': sys.version.split()[0], 'pandas': pd.__version__,
                   'numpy': np.__version__, 'scipy': scipy.__version__,
                   'statsmodels': statsmodels.__version__})
print('Semente aleatória:', SEMENTE)
''')
md(r'''
## 1. Dados: o que estamos medindo?
A **National Energy Administration (NEA)** divulga o indicador **全社会用电量**, consumo de eletricidade de toda a sociedade. Esta é uma fonte oficial adequada para **eletricidade**; consumo total de energia, incluindo combustíveis, é outro indicador, divulgado também pelo NBS.

- Unidade original: 100 milhões de kWh. Dividimos por 10 para obter **TWh** (bilhões de kWh).
- Frequência: **mensal**, com sazonalidade anual sugerida pelos dados. Um ano corresponde a 12 observações.
- Natureza: variável quantitativa contínua, de **fluxo**: soma da energia consumida durante o mês.
- Grade: janeiro/2015 a dezembro/2024, 120 meses.
- Coleta anterior: 99 valores diretamente publicados, 9 janeiros reconstituídos como bimestre menos fevereiro no mesmo boletim e 12 lacunas.

Os 12 meses sem valor separado na coleta são os dez dezembros e janeiro/fevereiro de 2024. Isso **não significa consumo zero**. Janeiro–fevereiro de 2024 totaliza 1.531,6 TWh, mas a divisão não foi localizada. Os acumulados de boletins diferentes apresentam pequenas divergências; não serão usados para reconstruir automaticamente meses nem para revelar valores apagados no experimento.

**Fontes:** [arquivo NEA](https://www.nea.gov.cn/sjzz/ghs/zjgx.htm), [bimestre de 2024](https://www.nea.gov.cn/2024-03/20/c_1310768302.htm), [total anual de 2024](https://www.nea.gov.cn/20250120/4f7f249bac714e7693adecac996d742f/c.html), [NBS: consumo total de energia](https://www.stats.gov.cn/english/PressRelease/202502/t20250228_1958822.html). Maio/2023 vem de [republicação que atribui os números à NEA](https://obor.nea.gov.cn/detail/19346.html). As URLs de cada valor estão no CSV original.
''')
code(r'''
dados = pd.read_csv(BASE / 'consumo_eletricidade_china_mensal.csv', parse_dates=['data']).set_index('data').sort_index()
serie = dados.valor_twh.asfreq('MS').copy()
diretos = dados.observado_direto_twh.notna().reindex(serie.index)
reconstituidos = dados.status.eq('reconstituido_mesma_fonte')
lacunas_originais = serie.isna()
assert serie.index.equals(pd.date_range('2015-01-01', '2024-12-01', freq='MS'))
assert serie.index.is_unique and serie.count() == 108
assert diretos.sum() == 99 and lacunas_originais.sum() == 12
assert (serie.dropna() > 0).all()
extraidos = pd.read_csv(BASE / 'valores_extraidos.csv')
anuais = extraidos[extraidos.tipo.eq('anual')].set_index('ano').valor_twh.sort_index()
print('SHA-256 do CSV:', hashlib.sha256((BASE / 'consumo_eletricidade_china_mensal.csv').read_bytes()).hexdigest())
tabela(dados[['valor_twh', 'status']], '01_serie_e_procedencia')
''')
md(r'''
### Estatística descritiva em palavras simples
A **média** é a soma dividida pela quantidade de valores; a **mediana** é o valor central depois de ordenar. Os **quartis** delimitam os 25%, 50% e 75% da distribuição. O **desvio-padrão** mede a dispersão em torno da média; a **variância** é seu quadrado. O **coeficiente de variação**, $100s/\bar y$, compara o desvio-padrão com a média em porcentagem. **Assimetria** indica desequilíbrio entre as caudas e **excesso de curtose** descreve a forma das caudas em relação à normal (referência zero).

Abaixo, usamos somente os 108 valores disponíveis, sem preencher nada. A soma é **parcial**, não o consumo de dez anos completos. As medidas misturam meses sazonais e anos com níveis diferentes: não são medidas de uma distribuição invariável no tempo.
''')
code(r'''
x = serie.dropna()
descritiva = pd.Series({'N disponível': x.count(), 'Média (TWh)': x.mean(), 'Mediana (TWh)': x.median(),
    'Desvio-padrão amostral (TWh)': x.std(ddof=1), 'Variância amostral (TWh²)': x.var(ddof=1),
    'Mínimo (TWh)': x.min(), 'Q1 (TWh)': x.quantile(.25), 'Q3 (TWh)': x.quantile(.75),
    'Máximo (TWh)': x.max(), 'Amplitude (TWh)': x.max()-x.min(), 'CV (%)': 100*x.std()/x.mean(),
    'Assimetria': x.skew(), 'Excesso de curtose': x.kurt(), 'Soma parcial (TWh)': x.sum()}, name='Valor').to_frame()
tabela(descritiva, '02_descritiva_original')
resumo_anual = serie.groupby(serie.index.year).agg(['count','mean','median','std'])
resumo_anual['Total anual NEA (TWh)'] = anuais
tabela(resumo_anual, '03_descritiva_anual')
fig, ax = plt.subplots()
ax.plot(serie, '.-', color='#176B87', label='Valores disponíveis; lacunas interrompem a linha')
ax.set(title='China — consumo de eletricidade, 2015–2024', ylabel='TWh/mês', xlabel='Mês')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=.2)
figura('01_serie_original')
''')
md(r'''
## 2. Eliminar observações e comparar métodos de reposição
**Imputar** significa estimar um valor que falta. Para saber se um método funciona, guardamos uma resposta conhecida, apagamos sua cópia de entrada e pedimos ao método para reconstruí-la. Só depois comparamos com a resposta guardada.

### Métodos comparados
| Método | Como funciona | Limitação |
|---|---|---|
| Média global | Coloca a média dos valores visíveis em cada lacuna. | Ignora crescimento e época do ano; referência simples. |
| Interpolação linear | Liga dois pontos visíveis por uma reta. | Pode apagar picos; nas pontas usamos o valor visível mais próximo. |
| PCHIP | Liga pontos por curvas cúbicas que preservam a forma local. | Não aprende o padrão anual; nas pontas também usa o valor mais próximo. |
| Regressão harmônica no log | Ajusta crescimento e ondas anuais aos valores visíveis. | Supõe crescimento aproximadamente exponencial e sazonalidade relativamente estável. |
| Harmônica + resíduo local | Ajusta o modelo anterior e interpola o que ele não explicou entre vizinhos. | Pode transportar choques locais para a lacuna. |

Para a interpolação linear, entre os meses $a$ e $b$:
$$\widehat y_t=y_a+\frac{t-a}{b-a}(y_b-y_a).$$
Usamos **posição mensal** (janeiro, fevereiro, março…), e não distância em dias.

Na regressão harmônica, modelamos:
$$\log y_t=\beta_0+\beta_1(t/12)+\sum_{k=1}^{3}\left[a_k\sin(2\pi kt/12)+b_k\cos(2\pi kt/12)\right]+e_t.$$
As ondas são chamadas de **termos de Fourier**: somadas, desenham um padrão que se repete a cada 12 meses. Três harmônicos permitem um desenho mais flexível que uma única onda, sem criar uma variável livre para dezembro, que nunca foi observado. Ajustamos os coeficientes por mínimos quadrados somente nos pontos visíveis. Aplicar a exponencial devolve valores positivos em TWh; é uma previsão central na escala original, sem correção de viés lognormal.

No método híbrido, a previsão é $\exp(\widehat{\log y_t}+\widetilde e_t)$, com $\widetilde e_t$ obtido por interpolação linear dos resíduos visíveis. Nas pontas, mantemos o último resíduo disponível.

### Regras do experimento
- Seleção dos métodos **somente em 2015–2022**: 2023 e 2024 ficam disponíveis para a comparação preditiva ao final.
- Apagamos apenas valores **diretamente publicados**; uma estimativa nunca é usada como resposta verdadeira.
- Cenários: 10% e 20% dos valores diretos removidos aleatoriamente; um bloco de 3 ou 6 meses consecutivos removido por vez.
- Cada cenário é repetido 20 vezes, com sementes fixas. Os métodos recebem exatamente as mesmas máscaras em cada repetição.
- Se fevereiro for apagado, retiramos também o janeiro reconstituído a partir daquele boletim, evitando dependência indireta do valor escondido. Esse janeiro adicional não entra no cálculo do erro.
- Não usamos totais anuais, acumulados ou valores apagados para ajustar os métodos.
- Trata-se de **reconstrução retrospectiva**: usar vizinhos posteriores ao buraco é permitido. Isso é diferente de prever o futuro.

### Como medir o erro?
$$MAE=\frac1n\sum |y_t-\widehat y_t|,\qquad RMSE=\sqrt{\frac1n\sum(y_t-\widehat y_t)^2}.$$
**MAE** é o erro absoluto médio em TWh. **RMSE** penaliza mais os erros grandes. Em ambos, menor é melhor. O **MAPE** expressa o erro absoluto relativo em porcentagem; pode ser problemático perto de zero, o que não ocorre aqui.
''')
code(r'''
METODOS = ['Média global', 'Linear', 'PCHIP', 'Harmônica log', 'Harmônica + resíduo']

def matriz_harmonica(n, k=3):
    t = np.arange(n, dtype=float)
    col = [np.ones(n), t/12]
    for j in range(1, k+1):
        col += [np.sin(2*np.pi*j*t/12), np.cos(2*np.pi*j*t/12)]
    return np.column_stack(col)

def imputar(s, metodo):
    s = s.astype(float).copy()
    ok = s.notna().to_numpy(); t = np.arange(len(s), dtype=float)
    assert ok.sum() >= 12 and (s.dropna() > 0).all()
    if metodo == 'Média global':
        estimativa = np.full(len(s), s.mean())
    elif metodo == 'Linear':
        estimativa = np.interp(t, t[ok], s.to_numpy()[ok])
    elif metodo == 'PCHIP':
        curva = PchipInterpolator(t[ok], s.to_numpy()[ok], extrapolate=False)
        estimativa = pd.Series(curva(t)).ffill().bfill().to_numpy()
    else:
        assert metodo in ['Harmônica log', 'Harmônica + resíduo']
        X = matriz_harmonica(len(s)); z = np.log(s.to_numpy()[ok])
        coef = np.linalg.lstsq(X[ok], z, rcond=None)[0]
        previsao_log = X @ coef
        if metodo == 'Harmônica + resíduo':
            residuo = z - previsao_log[ok]
            previsao_log += np.interp(t, t[ok], residuo)
        estimativa = np.exp(previsao_log)
    resultado = s.copy()
    resultado.iloc[np.flatnonzero(~ok)] = estimativa[~ok]
    assert resultado.notna().all() and (resultado > 0).all()
    np.testing.assert_array_equal(resultado[ok], s[ok])
    return resultado

def ocultar(s, datas):
    entrada = s.copy(); entrada.loc[datas] = np.nan
    for data in datas:
        janeiro = pd.Timestamp(data.year, 1, 1)
        if data.month == 2 and janeiro in entrada.index and reconstituidos.get(janeiro, False):
            entrada.loc[janeiro] = np.nan
    return entrada

def metricas(real, previsto):
    a = np.asarray(real, dtype=float); b = np.asarray(previsto, dtype=float)
    assert len(a) and np.isfinite(a).all() and np.isfinite(b).all()
    e = b-a
    return {'MAE': np.abs(e).mean(), 'RMSE': np.sqrt(np.mean(e**2)),
            'MAPE (%)': 100*np.mean(np.abs(e)/np.abs(a))}

periodo_experimento = serie.loc[:'2022-12-01']
elegiveis = diretos.reindex(periodo_experimento.index)

def sortear_mascara(s, elegivel, cenario, rng):
    candidatos = np.flatnonzero(elegivel.to_numpy())
    if cenario.startswith('Aleatório'):
        fracao = .1 if '10%' in cenario else .2
        pos = rng.choice(candidatos, size=max(1, round(fracao*len(candidatos))), replace=False)
    else:
        tamanho = 3 if '3 meses' in cenario else 6
        inicios = [i for i in range(len(s)-tamanho+1) if elegivel.iloc[i:i+tamanho].all()]
        inicio = rng.choice(inicios); pos = np.arange(inicio, inicio+tamanho)
    return s.index[np.sort(pos)]

CENARIOS = ['Aleatório 10%', 'Aleatório 20%', 'Bloco 3 meses', 'Bloco 6 meses']
registros = []; detalhes = []
for repeticao in range(20):
    for c, cenario in enumerate(CENARIOS):
        rng = np.random.default_rng(SEMENTE + 100*repeticao + c)
        apagadas = sortear_mascara(periodo_experimento, elegiveis, cenario, rng)
        entrada = ocultar(periodo_experimento, apagadas)
        assert entrada.loc[apagadas].isna().all()
        for metodo in METODOS:
            preenchida = imputar(entrada, metodo)
            medidas = metricas(periodo_experimento.loc[apagadas], preenchida.loc[apagadas])
            registros.append({'Repetição': repeticao, 'Cenário': cenario, 'Método': metodo,
                              'N apagado': len(apagadas), **medidas})
            for data in apagadas:
                detalhes.append({'Repetição': repeticao, 'Cenário': cenario, 'Método': metodo,
                    'Data': data, 'Real': periodo_experimento.loc[data], 'Estimado': preenchida.loc[data]})
experimentos = pd.DataFrame(registros)
experimentos.to_csv(SAIDA/'04_imputacao_todas_repeticoes.csv', index=False)
pd.DataFrame(detalhes).to_csv(SAIDA/'05_imputacao_valores_ocultados.csv', index=False)
comparacao = experimentos.groupby(['Cenário','Método']).agg(
    MAE_medio=('MAE','mean'), MAE_dp=('MAE','std'), RMSE_medio=('RMSE','mean'),
    MAPE_medio=('MAPE (%)','mean'), repeticoes=('MAE','count'))
tabela(comparacao, '06_imputacao_por_cenario')
ranking = comparacao.groupby('Método')[['MAE_medio','RMSE_medio','MAPE_medio']].mean().sort_values('MAE_medio')
melhor_imputador = ranking.index[0]
tabela(ranking, '07_ranking_imputacao')
melhores_cenario = comparacao.reset_index().sort_values('MAE_medio').groupby('Cenário').head(1).set_index('Cenário')
tabela(melhores_cenario, '08_melhores_por_cenario')
print('Vencedor pela média do MAE, com peso igual entre cenários:', melhor_imputador)
print('MAE_dp descreve variação entre máscaras; não é intervalo de confiança.')
''')
md(r'''
### Exibir as reposições, não apenas uma nota final
A tabela seguinte mostra um exemplo fixo de 20% removidos: valor verdadeiro, estimativa de cada método e erro absoluto do vencedor. A escolha do vencedor considera todos os cenários anteriores, não apenas esse exemplo.

O ranking usa média dos MAEs de cada cenário com **peso igual**. É um critério de comparação didática, não prova de superioridade universal. As máscaras se sobrepõem entre repetições; não são novas amostras independentes da economia chinesa.
''')
code(r'''
rng = np.random.default_rng(SEMENTE + 1)
apagadas_exemplo = sortear_mascara(periodo_experimento, elegiveis, 'Aleatório 20%', rng)
entrada_exemplo = ocultar(periodo_experimento, apagadas_exemplo)
exemplo = pd.DataFrame({'Real (TWh)': periodo_experimento.loc[apagadas_exemplo]})
for metodo in METODOS:
    exemplo[metodo] = imputar(entrada_exemplo, metodo).loc[apagadas_exemplo]
exemplo['Erro absoluto do vencedor'] = (exemplo[melhor_imputador]-exemplo['Real (TWh)']).abs()
tabela(exemplo, '09_exemplo_reposicoes')
fig, axs = plt.subplots(1,2,figsize=(13,4.3))
ranking.MAE_medio.sort_values().plot.barh(ax=axs[0], color='#176B87')
axs[0].set(title='Comparação de imputação', xlabel='MAE médio (TWh)', ylabel='')
axs[1].plot(periodo_experimento, color='gray', alpha=.6, label='Referência conhecida')
axs[1].scatter(apagadas_exemplo, exemplo['Real (TWh)'], marker='o', label='Valores ocultados')
axs[1].scatter(apagadas_exemplo, exemplo[melhor_imputador], marker='x', color='#D17820', label='Reposição vencedora')
axs[1].set(title=melhor_imputador, ylabel='TWh'); axs[1].legend(fontsize=8)
figura('02_imputacao')
''')
md(r'''
## 3. Série de trabalho: preencher as 12 lacunas originais
Agora ajustamos o método escolhido aos **108 valores disponíveis de toda a série** e preenchemos somente os 12 valores ausentes. Isso produz uma série de trabalho para análise retrospectiva, mantendo intactos os valores conhecidos.

**Limite do experimento:** nenhum dezembro estava disponível para ser ocultado. Assim, o erro observado nos testes não mede diretamente a qualidade das estimativas de dezembro. Mesmo um método que vence nos meses conhecidos depende de hipóteses para estimar um mês nunca observado. Também não impomos igualdade com acumulados de versões diferentes. A diferença para os agregados oficiais é apresentada como diagnóstico.

Criamos ainda uma alternativa por interpolação linear para verificar se as conclusões dos testes de estacionariedade mudam com o preenchimento. Imputação simples não propaga toda a incerteza: os testes seguintes devem ser lidos **condicionalmente ao preenchimento escolhido**.
''')
code(r'''
trabalho = imputar(serie, melhor_imputador)
alternativa_linear = imputar(serie, 'Linear')
procedencia = dados.status.copy()
procedencia.loc[lacunas_originais] = 'imputado_' + melhor_imputador
preenchimento = pd.DataFrame({'Original': serie, 'Trabalho (TWh)': trabalho,
                             'Alternativa linear': alternativa_linear, 'Procedência': procedencia})
tabela(preenchimento.loc[lacunas_originais], '10_lacunas_originais_preenchidas')
reconciliacao = pd.DataFrame({'Total NEA': anuais,
    'Soma série de trabalho': trabalho.groupby(trabalho.index.year).sum()})
reconciliacao['Diferença (TWh)'] = reconciliacao['Soma série de trabalho']-reconciliacao['Total NEA']
reconciliacao['Diferença (%)'] = 100*reconciliacao['Diferença (TWh)']/reconciliacao['Total NEA']
tabela(reconciliacao, '11_comparacao_agregados')
print('Bimestre jan–fev/2024 estimado:', round(trabalho.loc['2024-01':'2024-02'].sum(),2),
      'TWh; divulgado: 1531,6 TWh. Os valores não foram forçados a coincidir.')
''')
md(r'''
## 4. Estacionariedade: o comportamento estatístico muda com o tempo?
Uma série é **fracamente estacionária** quando sua média e sua variância são constantes ao longo do tempo e a relação entre dois valores depende da distância entre eles, não da data em si. Um consumo que cresce continuamente não oscila em torno de uma média fixa.

Não estacionariedade não significa “dados ruins”. Significa que certos modelos exigem transformações ou componentes explícitos para descrever a mudança. Holt e Holt-Winters, por exemplo, modelam nível, tendência e sazonalidade e não exigem que a série original seja estacionária.

### Dois testes com hipóteses opostas
| Teste | Hipótese nula ($H_0$) | Se p < 0,05 |
|---|---|---|
| ADF — Dickey-Fuller aumentado | Existe raiz unitária, uma forma de não estacionariedade. | Rejeitamos raiz unitária na especificação utilizada. |
| KPSS | A série é estacionária em torno do componente determinístico especificado. | Rejeitamos essa estacionariedade. |

**Raiz unitária**, em termos simples, é uma situação em que um choque pode deixar efeito persistente no nível. O ADF investiga se há evidência contra esse comportamento. O KPSS parte da hipótese oposta. “Não rejeitar” uma hipótese **não é provar** que ela é verdadeira.

Usamos nível de significância de 5%, fixado antes dos resultados. Com `c`, a referência é uma constante; com `ct`, admite-se uma tendência linear determinística. Estacionária **em torno de uma tendência** não significa estacionária em nível sem retirar essa tendência.

| Resultado conjunto | Leitura cuidadosa |
|---|---|
| ADF rejeita; KPSS não rejeita | Evidência compatível com estacionariedade na especificação. |
| ADF não rejeita; KPSS rejeita | Evidência compatível com não estacionariedade. |
| Ambos rejeitam | Resultados divergentes: investigar quebras, especificação e sazonalidade. |
| Nenhum rejeita | Evidência inconclusiva; os testes podem ter pouco poder. |

### Transformações comparadas
- **Logaritmo:** $z_t=\log y_t$. Coloca mudanças proporcionais em escala comparável; não garante estacionariedade.
- **Primeira diferença:** $\Delta z_t=z_t-z_{t-1}$. Aproxima a taxa mensal de crescimento; multiplique por 100 para porcentagem aproximada.
- **Diferença sazonal:** $\Delta_{12}z_t=z_t-z_{t-12}$. Compara o mesmo mês de anos consecutivos.
- **Diferença comum e sazonal:** $\Delta\Delta_{12}z_t$. Combina as duas operações. Diferenciar demais também pode prejudicar o modelo; não se escolhe uma transformação apenas para obter p pequeno.

No ADF, o número de defasagens é escolhido por AIC entre 0 e 12. No KPSS, usamos a seleção automática da biblioteca. Apresentamos ambos os números. Os p-valores tabelados do KPSS são limitados: `≤0,01` e `≥0,10` indicam limites, não valores exatos.

**Referências:** [ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html), [KPSS](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html). Esses testes não são testes específicos de raiz unitária sazonal nem substituem o exame dos gráficos.
''')
code(r'''
def leitura_testes(p_adf, p_kpss):
    a, k = p_adf < .05, p_kpss < .05
    if a and not k: return 'Compatível com estacionariedade na especificação'
    if not a and k: return 'Compatível com não estacionariedade'
    if a and k: return 'Testes divergentes: investigar especificação'
    return 'Inconclusivo'

def testar(s, identificacao):
    z = np.log(s)
    versoes = [('Nível', s, 'c'), ('Log nível', z, 'c'), ('Log com tendência', z, 'ct'),
               ('Diferença log', z.diff(), 'c'), ('Diferença sazonal log', z.diff(12), 'c'),
               ('Diferença comum + sazonal log', z.diff(12).diff(), 'c')]
    linhas = []
    for nome, v, reg in versoes:
        v = v.dropna()  # Apenas perdas iniciais causadas por diferenças; calendário já está completo.
        opcao_adf = {'result_object': False} if 'result_object' in inspect.signature(adfuller).parameters else {}
        adf = adfuller(v, maxlag=12, regression=reg, autolag='AIC', **opcao_adf)
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter('always')
            opcao_kpss = {'result_object': False} if 'result_object' in inspect.signature(kpss).parameters else {}
            kp = kpss(v, regression=reg, nlags='auto', **opcao_kpss)
        p_txt = '≤0,01' if kp[1] <= .01 else ('≥0,10' if kp[1] >= .1 else f'{kp[1]:.4f}')
        linhas.append({'Preenchimento': identificacao, 'Transformação': nome, 'Regressão': reg,
            'N': len(v), 'ADF estatística': adf[0], 'ADF p': adf[1], 'ADF lags': adf[2],
            'ADF crítico 5%': adf[4]['5%'], 'KPSS estatística': kp[0], 'KPSS p': p_txt,
            'KPSS lags': kp[2], 'KPSS crítico 5%': kp[3]['5%'],
            'Leitura a 5%': leitura_testes(adf[1], kp[1]),
            'Aviso KPSS': ' | '.join(str(w.message) for w in avisos)})
    return pd.DataFrame(linhas)

testes = pd.concat([testar(trabalho, melhor_imputador), testar(alternativa_linear, 'Linear (sensibilidade)')], ignore_index=True)
tabela(testes.drop(columns='Aviso KPSS'), '12_testes_estacionariedade')
testes.to_csv(SAIDA/'12_testes_com_avisos.csv', index=False)
print('Leitura principal em log nível:', testes.iloc[1]['Leitura a 5%'])
print('Compare as mesmas transformações entre os dois preenchimentos antes de concluir.')
principal_est = testes[testes.Preenchimento.eq(melhor_imputador)].set_index('Transformação')
r_nivel = principal_est.loc['Log nível']
r_dif = principal_est.loc['Diferença sazonal log']
display(Markdown(f"**Neste conjunto:** no log em nível, ADF p = {r_nivel['ADF p']:.4f} e KPSS p {r_nivel['KPSS p']}. Os dois apontam para não estacionariedade em nível. Para a diferença sazonal do log, ADF p = {r_dif['ADF p']:.6f} e KPSS p {r_dif['KPSS p']}: o resultado é compatível com estacionariedade na especificação. A primeira diferença também passa nos critérios, e a inclusão apenas de tendência deixa os testes inconclusivos. As mesmas leituras qualitativas aparecem com preenchimento linear. **Uma escolha didática útil é estudar a diferença de 12 meses do log**, que compara cada mês ao mesmo mês do ano anterior; não é necessário acrescentar outra diferença somente para diminuir o p-valor."))
fig, axs = plt.subplots(3,1,figsize=(12,7),sharex=True)
for ax, s, titulo in zip(axs, [np.log(trabalho), np.log(trabalho).diff(), np.log(trabalho).diff(12)],
                        ['Log do consumo', 'Primeira diferença do log', 'Diferença de 12 meses do log']):
    ax.plot(s, color='#176B87'); ax.set_title(titulo); ax.grid(alpha=.2)
figura('03_transformacoes_estacionariedade')
''')
md(r'''
## 5. Outliers: valores estranhos em relação ao comportamento esperado
Um **outlier** é um valor muito distante do padrão usado como referência. Um pico no verão pode ser normal; uma alta inesperada no meio de um período estável pode merecer investigação. Por isso, não aplicamos simplesmente um limite único ao consumo de toda a década.

### Detecção: STL robusta + mediana e MAD dos resíduos
A **STL** separa a série em tendência, sazonalidade e resíduo usando regressões locais suaves (**LOESS**). “Robusta” significa reduzir a influência de observações que se afastam muito do ajuste. Usamos o log do consumo, período 12, janela sazonal 13 e tendência 25 meses, fixados para o exercício.

$$\log y_t=T_t+S_t+r_t.$$
Nos resíduos, calculamos a mediana $m$ e o **MAD**, desvio absoluto mediano:
$$MAD=\operatorname{mediana}(|r_t-m|),\quad \widehat\sigma_r=1{,}4826\,MAD,\quad z_t^*=\frac{r_t-m}{\widehat\sigma_r}.$$
O fator 1,4826 torna a escala comparável ao desvio-padrão sob normalidade. Mediana e MAD sofrem menos influência de extremos que média e desvio-padrão.

Sinalizamos $|z_t^*|>3{,}5$. É uma **regra exploratória**, não um teste com falso positivo controlado. Pontos preenchidos e janeiros reconstituídos não são classificados como outliers observados; entram no ajuste, mas a sinalização e a escala usam apenas os valores diretamente publicados. Nas extremidades e perto de lacunas, a interpretação é ainda mais incerta.

### Tratamento dos suspeitos originais
Sem prova de erro de registro, preservamos os números oficiais. Para cumprir o exercício, criamos uma **cópia tratada por winsorização dos resíduos**: um resíduo além do limite é trazido até o limite mais próximo. Isso altera menos o ponto que substituí-lo inteiramente pela tendência e sazonalidade. É uma análise de sensibilidade, **não uma correção factual da NEA**.

[Referência da STL](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html).
''')
code(r'''
def detectar(s, elegivel):
    elegivel = elegivel.reindex(s.index).fillna(False).astype(bool)
    decom = STL(np.log(s), period=12, seasonal=13, trend=25, robust=True).fit()
    residuo = pd.Series(decom.resid, index=s.index)
    m = residuo[elegivel].median()
    mad = (residuo[elegivel]-m).abs().median()
    escala = max(1.4826*mad, 1e-8)
    z = (residuo-m)/escala
    flags = z.abs().gt(3.5) & elegivel
    esperado = pd.Series(decom.trend+decom.seasonal, index=s.index)
    return {'flags': flags, 'residuo': residuo, 'mediana': m, 'escala': escala,
            'z': z, 'esperado': esperado, 'stl': decom}

def tratar(s, deteccao, metodo):
    f = deteccao['flags']; saida = s.copy()
    m, escala = deteccao['mediana'], deteccao['escala']
    if metodo == 'Sem tratamento': return saida
    if metodo == 'Winsorizar resíduos':
        r = deteccao['residuo'].clip(m-3.5*escala, m+3.5*escala)
        estimado = np.exp(deteccao['esperado']+r)
    elif metodo == 'Reconstrução STL':
        estimado = np.exp(deteccao['esperado']+m)
    elif metodo == 'Remover e imputar':
        estimado = imputar(s.mask(f), melhor_imputador)
    else: raise ValueError(metodo)
    saida.loc[f] = estimado.loc[f]
    np.testing.assert_array_equal(saida[~f], s[~f])
    return saida

deteccao_original = detectar(trabalho, diretos)
suspeitos = deteccao_original['flags']
tratada = tratar(trabalho, deteccao_original, 'Winsorizar resíduos')
quadro_suspeitos = pd.DataFrame({'Original disponível': serie, 'Escore robusto': deteccao_original['z'],
                               'Cópia tratada (TWh)': tratada})
quadro_suspeitos['Alteração (TWh)'] = tratada-serie
tabela(quadro_suspeitos.loc[suspeitos], '13_outliers_suspeitos_e_tratamento')
print('Suspeitos diretamente observados:', int(suspeitos.sum()))
print('Esses sinais não comprovam erros. As estimativas de lacunas podem influenciar a detecção.')
fig, ax = plt.subplots()
ax.plot(trabalho, label='Série de trabalho', color='#176B87')
ax.scatter(trabalho.index[suspeitos], trabalho[suspeitos], color='#BA3636', label='Suspeitos', zorder=4)
ax.scatter(tratada.index[suspeitos], tratada[suspeitos], marker='x', color='#D17820', label='Cópia winsorizada', zorder=5)
ax.set(title='Detecção com tendência e sazonalidade consideradas', ylabel='TWh'); ax.legend()
figura('04_outliers_originais')
''')
md(r'''
### Experimento controlado com outliers artificiais
Os suspeitos reais não têm uma “resposta correta” alternativa conhecida. Por isso, **também criamos erros artificiais**, mesmo se houver suspeitos reais. Se nenhum ponto fosse sinalizado, esse experimento ainda permitiria cumprir a atividade.

Selecionamos seis valores diretamente publicados, fora da lista de suspeitos e longe das pontas. Guardamos seus valores, multiplicamos três por fatores acima de 2 e três por fatores abaixo de 0,4, e executamos novamente o detector **sem informar os locais das alterações**.

Comparamos quatro opções: não tratar; winsorizar resíduos; reconstruir o componente esperado pela STL; remover os pontos detectados e imputar. O tratamento só age onde o detector sinaliza: um outlier artificial não detectado continua prejudicando o resultado.

**Precisão** = fração dos sinais do detector que coincide com uma alteração artificial. **Sensibilidade (recall)** = fração das seis alterações encontradas. Para essa avaliação, excluímos os suspeitos já existentes, pois não sabemos se eram erros. Um “falso positivo” aqui é um alarme fora das seis alterações no domínio avaliado, não uma afirmação de que o valor oficial é correto.

Além do erro nas seis posições, medimos alterações desnecessárias nos outros pontos avaliáveis, para não premiar um método que conserta alguns valores e distorce muitos outros. Este é um cenário fixo de erros grandes e isolados; resultados podem mudar para erros menores, blocos ou mudanças permanentes de nível.
''')
code(r'''
rng = np.random.default_rng(SEMENTE+9000)
dominio = diretos & ~suspeitos
candidatos = trabalho.index[dominio & (trabalho.index.year >= 2016) & (trabalho.index.year <= 2023)]
injetados = pd.DatetimeIndex(sorted(rng.choice(candidatos, size=6, replace=False)))
fatores = np.array([2.5, .30, 2.2, .35, 2.8, .25])
contaminada = trabalho.copy(); contaminada.loc[injetados] *= fatores
artificial = pd.Series(False, index=trabalho.index); artificial.loc[injetados] = True
det_artificial = detectar(contaminada, diretos)
marcados = det_artificial['flags']
tp = int((marcados & artificial & dominio).sum())
fp = int((marcados & ~artificial & dominio).sum())
fn = int((~marcados & artificial & dominio).sum())
print({'Alterações artificiais': 6, 'Verdadeiros positivos': tp, 'Falsos positivos no domínio': fp,
       'Falsos negativos': fn, 'Precisão': tp/(tp+fp) if tp+fp else np.nan, 'Recall': tp/6})
metodos_outliers = ['Sem tratamento', 'Winsorizar resíduos', 'Reconstrução STL', 'Remover e imputar']
resultados_outliers = {}; linhas = []
for nome in metodos_outliers:
    corrigida = tratar(contaminada, det_artificial, nome)
    resultados_outliers[nome] = corrigida
    medidas = metricas(trabalho.loc[injetados], corrigida.loc[injetados])
    intactos = dominio & ~artificial
    linhas.append({'Método': nome, **medidas,
        'MAE demais pontos': (corrigida[intactos]-trabalho[intactos]).abs().mean(),
        'MAE todo domínio': (corrigida[dominio]-trabalho[dominio]).abs().mean()})
comparacao_outliers = pd.DataFrame(linhas).set_index('Método').sort_values('MAE')
tabela(comparacao_outliers, '14_comparacao_tratamentos_outliers')
melhor_outlier = comparacao_outliers.index[0]
ex_out = pd.DataFrame({'Referência': trabalho.loc[injetados], 'Fator': fatores,
                      'Contaminado': contaminada.loc[injetados], 'Detectado': marcados.loc[injetados]})
for nome, s in resultados_outliers.items(): ex_out[nome] = s.loc[injetados]
tabela(ex_out, '15_outliers_artificiais_valores')
fig, ax = plt.subplots()
ax.plot(trabalho, color='gray', label='Referência preservada')
ax.scatter(injetados, contaminada.loc[injetados], color='#BA3636', label='Erros artificiais')
ax.scatter(injetados, resultados_outliers[melhor_outlier].loc[injetados], marker='x', color='#176B87', label=melhor_outlier)
ax.set(title='Experimento de tratamento de erros artificiais', ylabel='TWh'); ax.legend(fontsize=9)
figura('05_outliers_artificiais')
''')
md(r'''
## 6. Média móvel, tendência, sazonalidade e ciclo
### Média móvel
Uma média móvel substitui cada ponto pela média de uma janela próxima. Ela reduz oscilações curtas e ajuda a enxergar o movimento geral.

A média móvel **trailing** de $k$ meses usa o mês atual e os $k-1$ anteriores:
$$MM_k(t)=\frac1k\sum_{j=0}^{k-1}y_{t-j}.$$
Com $k=3$, a média em março usa janeiro, fevereiro e março. Janelas maiores suavizam mais e respondem mais lentamente a mudanças. Exigimos janelas completas: os primeiros $k-1$ resultados ficam ausentes.

A média móvel **centrada 2×12**, comum em dados mensais, primeiro calcula médias de 12 meses e depois centraliza com uma média de duas delas. Seus pesos são 1/24 nas pontas $t-6$ e $t+6$ e 1/12 nos onze pontos interiores. Ela usa observações futuras: serve para descrição histórica, não para previsão em tempo real.

Abaixo, “cópia tratada” significa apenas o tratamento didático dos suspeitos; os outliers artificiais **não entram** nos cálculos finais. Como essa série foi preenchida e tratada retrospectivamente, até a média trailing calculada nela não representa uma informação histórica disponível em tempo real.
''')
code(r'''
movel = pd.DataFrame({'Trabalho': trabalho, 'Cópia tratada': tratada})
for k in [3,6,12]: movel[f'MM{k}'] = tratada.rolling(k, min_periods=k).mean()
pesos = np.r_[.5, np.ones(11), .5]/12
movel['MM centrada 2x12'] = tratada.rolling(13, center=True).apply(lambda v: np.dot(v,pesos), raw=True)
assert np.isclose(pesos.sum(), 1)
assert movel.MM12.isna().sum() == 11 and movel['MM centrada 2x12'].isna().sum() == 12
tabela(movel.tail(18), '16_medias_moveis_amostra')
movel.to_csv(SAIDA/'16_medias_moveis_completas.csv')
fig, ax = plt.subplots()
ax.plot(tratada, color='gray', alpha=.5, label='Cópia tratada')
for col in ['MM3','MM6','MM12','MM centrada 2x12']: ax.plot(movel[col], label=col)
ax.set(title='Médias móveis: suavização e atraso', ylabel='TWh'); ax.legend(ncol=3,fontsize=9)
figura('06_medias_moveis')
''')
md(r'''
### Quatro componentes diferentes
| Componente | Significado | Exemplo intuitivo |
|---|---|---|
| Tendência ($T$) | Movimento de longo prazo. | Crescimento do consumo ao longo dos anos. |
| Sazonalidade ($S$) | Padrão ligado ao calendário, com período regular. | Consumo maior em julho/agosto. |
| Ciclo ($C$) | Oscilação mais lenta em torno da tendência, sem período fixo obrigatório. | Expansões e desacelerações prolongadas. |
| Resíduo ($R$) | Parte não explicada pelos componentes escolhidos. | Choques, ruído, erros e efeitos não modelados. |

**Ciclo não é sinônimo de sazonalidade nem de resíduo.** Sua separação da tendência depende do método; não existe uma única decomposição verdadeira observável.

Usamos STL robusta em **TWh**, nesta seção, para manter uma decomposição aditiva de interpretação simples:
$$y_t=TC_t+S_t+R_t.$$
A STL produz uma componente suave que chamamos de **tendência-ciclo ($TC$)**. Para separar movimentos mais longos, aplicamos o filtro **Hodrick–Prescott (HP)** a $TC$:
$$TC_t=T_t+C_t,\qquad y_t=T_t+C_t+S_t+R_t.$$

O HP procura uma tendência próxima aos dados, mas penaliza mudanças bruscas em sua inclinação:
$$\min_T\sum_t(TC_t-T_t)^2+\lambda\sum_t[(T_{t+1}-T_t)-(T_t-T_{t-1})]^2.$$
Quanto maior $\lambda$, mais lisa a tendência. Usamos **129.600**, convenção mensal derivada de $1600(12/4)^4$. Aplicá-lo à componente suave da STL deixa no ciclo apenas variações suaves; o resíduo irregular permanece separado. Essa combinação é uma **escolha descritiva**, não uma identificação comprovada do ciclo econômico da China.

O HP pode produzir oscilações artificiais e é sensível às extremidades e a $\lambda$. Exibimos uma comparação com metade e o dobro do parâmetro. Tanto a STL quanto o HP usam a amostra histórica completa. A sazonalidade de dezembro depende especialmente da imputação, pois esse mês não foi observado.

[Referência do HP e da escolha mensal de $\lambda$](https://www.statsmodels.org/stable/generated/statsmodels.tsa.filters.hp_filter.hpfilter.html).
''')
code(r'''
dec = STL(tratada, period=12, seasonal=13, trend=25, robust=True).fit()
tc = pd.Series(dec.trend, index=tratada.index)
ciclo, tendencia = hpfilter(tc, lamb=129600)
componentes = pd.DataFrame({'Consumo tratado': tratada, 'Tendência': tendencia, 'Ciclo': ciclo,
                           'Sazonalidade': dec.seasonal, 'Resíduo': dec.resid, 'Tendência-ciclo STL': tc})
reconstruida = componentes[['Tendência','Ciclo','Sazonalidade','Resíduo']].sum(axis=1)
np.testing.assert_allclose(reconstruida, tratada, atol=1e-8)
tabela(componentes.tail(18), '17_componentes_amostra')
componentes.to_csv(SAIDA/'17_componentes_completos.csv')
perfil_sazonal = componentes.Sazonalidade.groupby(componentes.index.month).agg(['mean','min','max'])
perfil_sazonal['Observações mensais diretas'] = diretos.groupby(diretos.index.month).sum()
tabela(perfil_sazonal, '18_sazonalidade_por_mes')
fig, axs = plt.subplots(5,1,figsize=(12,10),sharex=True)
for ax, col in zip(axs, ['Consumo tratado','Tendência','Sazonalidade','Ciclo','Resíduo']):
    ax.plot(componentes[col], color='#176B87'); ax.set_ylabel('TWh'); ax.set_title(col); ax.grid(alpha=.2)
figura('07_decomposicao')

sensibilidade_ciclo = pd.DataFrame(index=tratada.index)
for lam in [64800,129600,259200]:
    c, t = hpfilter(tc, lamb=lam); sensibilidade_ciclo[str(lam)] = c
sensibilidade_ciclo.to_csv(SAIDA/'19_sensibilidade_ciclo.csv')
ax = sensibilidade_ciclo.plot(title='Ciclo estimado: sensibilidade ao parâmetro do HP', ylabel='TWh')
figura('08_sensibilidade_ciclo')
''')
md(r'''
## 7. Suavização exponencial: Holt e Holt-Winters
**Suavizar** é atualizar uma estimativa do comportamento da série, dando pesos aos dados. Na suavização exponencial, a influência das observações antigas diminui progressivamente. Isso difere de uma média móvel, que descarta totalmente valores fora da janela.

### Holt: nível e tendência
Holt atualiza um **nível** $\ell_t$ (patamar atual) e uma **tendência** $b_t$ (mudança por mês):
$$\ell_t=\alpha y_t+(1-\alpha)(\ell_{t-1}+b_{t-1}),$$
$$b_t=\beta(\ell_t-\ell_{t-1})+(1-\beta)b_{t-1},\qquad \widehat y_{t+h}=\ell_t+h b_t.$$
$\alpha$ controla a velocidade de atualização do nível; $\beta$, a da tendência. Valores altos reagem mais rapidamente. Valores baixos produzem mais estabilidade. **Holt não inclui sazonalidade**, portanto é uma comparação importante para esta série.

### Holt-Winters: nível, tendência e sazonalidade
Holt-Winters acrescenta $s_t$, um fator ou desvio sazonal, com período $m=12$.

**Versão aditiva:** adequada quando a amplitude sazonal é aproximadamente constante em TWh.
$$\ell_t=\alpha(y_t-s_{t-m})+(1-\alpha)(\ell_{t-1}+b_{t-1}),$$
$$b_t=\beta(\ell_t-\ell_{t-1})+(1-\beta)b_{t-1},$$
$$s_t=\gamma(y_t-\ell_{t-1}-b_{t-1})+(1-\gamma)s_{t-m},$$
$$\widehat y_{t+h}=\ell_t+h b_t+s_{t+h-m(k+1)},\quad k=\lfloor(h-1)/m\rfloor.$$
A equação sazonal acima usa o nível previsto no passo anterior, convenção da implementação utilizada.

**Versão multiplicativa:** adequada quando a amplitude cresce proporcionalmente ao nível; exige valores positivos.
$$\ell_t=\alpha\frac{y_t}{s_{t-m}}+(1-\alpha)(\ell_{t-1}+b_{t-1}),$$
$$b_t=\beta(\ell_t-\ell_{t-1})+(1-\beta)b_{t-1},$$
$$s_t=\gamma\frac{y_t}{\ell_{t-1}+b_{t-1}}+(1-\gamma)s_{t-m},$$
$$\widehat y_{t+h}=(\ell_t+h b_t)\,s_{t+h-m(k+1)}.$$
Um fator sazonal de 1,10 representa 10% acima do nível previsto. $\gamma$ controla a atualização da sazonalidade. Nas duas versões usadas aqui a **tendência é aditiva e não amortecida**; “multiplicativa” refere-se ao componente sazonal.

Os parâmetros e os estados iniciais são estimados numericamente por mínimos quadrados não lineares (`least_squares`) para reduzir o erro de ajuste. A tabela exibirá $\alpha$, $\beta$ e $\gamma$, e eventuais avisos do otimizador. Um parâmetro próximo de zero pode indicar um componente quase fixo, não ausência obrigatória dele.

**Referências:** [Holt](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.Holt.html), [Holt-Winters](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html).

### Avaliação temporal: prever sem olhar o futuro
1. **Treino 2015–2022 → validação 2023:** comparamos Holt, Holt-Winters aditivo, Holt-Winters multiplicativo e uma referência sazonal ingênua ($\widehat y_t=y_{t-12}$). Escolhemos o menor MAE de validação.
2. **Treino 2015–2023 → teste 2024:** reestimamos cada modelo com o período ampliado e mostramos o resultado, mantendo a escolha feita em 2023. Não escolhemos novamente pelo teste.
3. Em cada treino, o preenchimento e o tratamento de suspeitos são **refeitos só com dados daquele treino**. Não utilizamos a série completada com toda a década para prever 2023 ou 2024.
4. O erro é calculado somente nos meses de validação/teste **diretamente publicados**. Janeiro reconstituído e meses ausentes são excluídos da avaliação. O teste de 2024 cobre março–novembro, não o ano completo.

A referência sazonal pode usar um mês imputado no treino; isso é identificado como limitação. A hipótese didática é que as publicações do treino já estavam disponíveis no instante da previsão. Esta não é uma simulação exata de tempo real: não modelamos atrasos de publicação nem revisões de cada versão histórica.
''')
code(r'''
MODELOS = ['Sazonal ingênuo', 'Holt', 'Holt-Winters aditivo', 'Holt-Winters multiplicativo']
avisos_modelos = []

def preparar_treino(s):
    # Não acessa datas posteriores ao fim de s.
    completa = imputar(s, melhor_imputador)
    det = detectar(completa, diretos.reindex(s.index))
    return tratar(completa, det, 'Winsorizar resíduos')

def ajustar_prever(s, nome, horizonte=12, etapa=''):
    if nome == 'Sazonal ingênuo':
        ind = pd.date_range(s.index[-1]+pd.offsets.MonthBegin(), periods=horizonte, freq='MS')
        return None, pd.Series(np.resize(s.iloc[-12:].to_numpy(), horizonte), index=ind)
    # Escala numérica menor facilita a otimização; saída é reconvertida a TWh.
    y = s/1000
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        if nome == 'Holt':
            fit = Holt(y, initialization_method='estimated').fit(optimized=True, method='least_squares')
        else:
            saz = 'add' if nome.endswith('aditivo') else 'mul'
            fit = ExponentialSmoothing(y, trend='add', seasonal=saz, seasonal_periods=12,
                initialization_method='estimated').fit(optimized=True, use_brute=True, method='least_squares')
    sucesso = getattr(fit, 'mle_retvals', {}).get('success', True)
    for w in ws: avisos_modelos.append({'Etapa': etapa, 'Modelo': nome, 'Aviso': str(w.message)})
    if not sucesso: avisos_modelos.append({'Etapa': etapa, 'Modelo': nome, 'Aviso': 'Otimizador não declarou convergência.'})
    return fit, fit.forecast(horizonte)*1000

avaliacoes = []; previsoes = {}; ajustes = {}; parametros = []
for fim, ano, etapa in [('2022-12-01',2023,'Validação'), ('2023-12-01',2024,'Teste')]:
    treino = preparar_treino(serie.loc[:fim])
    alvo = serie.loc[str(ano)]; elegivel = diretos.loc[str(ano)]
    for nome in MODELOS:
        fit, pred = ajustar_prever(treino, nome, etapa=etapa)
        assert pred.index.equals(alvo.index)
        resultado = metricas(alvo[elegivel], pred[elegivel])
        avaliacoes.append({'Etapa': etapa, 'Modelo': nome, 'N avaliado': int(elegivel.sum()), **resultado})
        previsoes[(etapa,nome)] = pred; ajustes[(etapa,nome)] = fit
        if fit is not None:
            parametros.append({'Etapa': etapa, 'Modelo': nome,
                'alpha': fit.params['smoothing_level'], 'beta': fit.params['smoothing_trend'],
                'gamma': fit.params.get('smoothing_seasonal', np.nan)})
avaliacoes = pd.DataFrame(avaliacoes)
validacao = avaliacoes[avaliacoes.Etapa.eq('Validação')].set_index('Modelo').sort_values('MAE')
modelo_escolhido = validacao.index[0]
teste = avaliacoes[avaliacoes.Etapa.eq('Teste')].set_index('Modelo')
teste['Escolhido na validação'] = teste.index == modelo_escolhido
tabela(validacao, '20_comparacao_validacao_2023')
tabela(teste, '21_comparacao_teste_2024')
tabela(pd.DataFrame(parametros).set_index(['Etapa','Modelo']), '22_parametros_suavizacao', casas=6)
print('Modelo escolhido antes de examinar o teste:', modelo_escolhido)
print('O ranking de teste não redefine a escolha.')
melhor_no_teste = teste.MAE.idxmin()
display(Markdown(f"**Leitura da comparação:** {modelo_escolhido} foi escolhido pela validação. No teste de 2024, o menor MAE descritivo foi de {melhor_no_teste} ({teste.loc[melhor_no_teste, 'MAE']:.2f} TWh). Mostrar esse resultado é informativo, mas trocar a escolha depois de olhar o teste transformaria o teste em mais uma etapa de seleção. O resultado oficial do procedimento continua sendo o do modelo escolhido em 2023."))

quadro_previsoes = pd.DataFrame({'Original 2024': serie.loc['2024'], 'Avaliado': diretos.loc['2024']})
for nome in MODELOS: quadro_previsoes[nome] = previsoes[('Teste',nome)]
tabela(quadro_previsoes, '23_previsoes_2024')
fig, ax = plt.subplots()
ax.plot(serie.loc['2023':], 'o-', color='black', label='Série disponível', alpha=.7)
for nome in MODELOS: ax.plot(previsoes[('Teste',nome)], '--', label=nome)
ax.axvline(pd.Timestamp('2024-01-01'), color='gray', linestyle=':')
ax.set(title='Previsões de 2024 feitas usando somente treino até 2023', ylabel='TWh')
ax.legend(fontsize=8,ncol=2)
figura('09_previsoes_teste')
''')
md(r'''
### Ajustes suavizados e estados estimados na década completa
Por fim, ajustamos Holt e as duas versões de Holt-Winters à cópia tratada de toda a década para mostrar os valores ajustados, o nível, a tendência e a sazonalidade estimados. Essa é uma visualização retrospectiva; seu erro dentro da amostra não substitui o teste temporal anterior.

O valor ajustado é uma previsão de um passo à frente usando os estados anteriores do modelo. Os parâmetros, porém, foram estimados usando a amostra completa. O nível é medido em TWh e a tendência em TWh por mês. A sazonalidade aditiva está em TWh; a multiplicativa é um fator sem unidade.
''')
code(r'''
ajustes_completos = pd.DataFrame({'Série de trabalho': trabalho, 'Cópia tratada': tratada})
estados = pd.DataFrame(index=tratada.index)
parametros_finais = []
for nome in MODELOS[1:]:
    fit, _ = ajustar_prever(tratada, nome, etapa='Ajuste descritivo completo')
    ajustes_completos[nome] = fit.fittedvalues*1000
    estados[nome+' | nível TWh'] = fit.level*1000
    estados[nome+' | tendência TWh/mês'] = fit.trend*1000
    if 'Winters' in nome:
        estados[nome+' | sazonalidade'] = fit.season * (1000 if nome.endswith('aditivo') else 1)
    parametros_finais.append({'Modelo': nome, 'alpha': fit.params['smoothing_level'],
        'beta': fit.params['smoothing_trend'], 'gamma': fit.params.get('smoothing_seasonal',np.nan)})
tabela(pd.DataFrame(parametros_finais).set_index('Modelo'), '24_parametros_ajuste_completo', casas=6)
tabela(estados.tail(12), '25_estados_finais')
estados.to_csv(SAIDA/'25_estados_completos.csv')
ajustes_completos.to_csv(SAIDA/'26_ajustes_completos.csv')
fig, axs = plt.subplots(3,1,figsize=(12,8),sharex=True)
for ax, nome in zip(axs, MODELOS[1:]):
    ax.plot(tratada, color='gray', alpha=.6, label='Cópia tratada')
    ax.plot(ajustes_completos[nome], label=nome, color='#176B87')
    ax.set_ylabel('TWh'); ax.legend(fontsize=9)
figura('10_ajustes_suavizados')
pd.DataFrame(avisos_modelos, columns=['Etapa','Modelo','Aviso']).to_csv(SAIDA/'27_avisos_modelos.csv', index=False)
if avisos_modelos:
    tabela(pd.DataFrame(avisos_modelos), '27_avisos_modelos')
else:
    print('Ajustes sem avisos; otimizadores declararam convergência.')
''')
md(r'''
## 8. Conclusões: o que os cálculos permitem afirmar?
O resumo abaixo é gerado a partir dos resultados efetivamente calculados. Isso evita escrever antecipadamente que um método “deveria” vencer. Um vencedor no experimento de imputação não é necessariamente o vencedor de previsão: são tarefas diferentes.
''')
code(r'''
principal = testes[testes.Preenchimento.eq(melhor_imputador)]
mae_imp = ranking.loc[melhor_imputador,'MAE_medio']
mae_val = validacao.loc[modelo_escolhido,'MAE']
mae_teste = teste.loc[modelo_escolhido,'MAE']
texto = f"""**Imputação:** {melhor_imputador} teve o menor MAE médio nos cenários avaliados: **{mae_imp:.2f} TWh**. Isso não valida automaticamente dezembro, que nunca foi observado nesta coleta.

**Estacionariedade:** no log em nível, a leitura conjunta foi: **{principal.iloc[1]['Leitura a 5%']}**. Compare a tabela de diferenças e a alternativa linear; os testes não incorporam toda a incerteza da imputação.

**Outliers:** foram sinalizados **{int(suspeitos.sum())} valores diretamente publicados** como suspeitos, preservados na base oficial e winsorizados apenas em uma cópia de estudo. No experimento artificial, o detector encontrou **{tp} de 6** alterações. O menor MAE nas posições injetadas foi de **{melhor_outlier}**; confira também o erro nos demais pontos.

**Componentes:** a média móvel torna o movimento geral mais visível. A STL separa sazonalidade e resíduo; o HP divide a componente suave em tendência e ciclo estimados. A soma dos quatro componentes foi conferida numericamente. O ciclo é dependente do filtro, não uma prova de recessões ou expansões.

**Previsão:** a escolha pelo MAE de 2023 foi **{modelo_escolhido}** ({mae_val:.2f} TWh). Mantida essa escolha, o MAE nos **9 meses diretamente observados de 2024** foi **{mae_teste:.2f} TWh**. Esse número não representa o erro nos 12 meses de 2024.
"""
display(Markdown(texto))
(SAIDA/'conclusoes.md').write_text(texto)
base_final = pd.DataFrame({'Original disponível TWh': serie, 'Diretamente observado': diretos,
    'Lacuna original': lacunas_originais, 'Procedência': procedencia,
    'Série de trabalho TWh': trabalho, 'Suspeito observado': suspeitos,
    'Cópia tratada TWh': tratada})
base_final.to_csv(SAIDA/'28_base_final_com_mascaras.csv')
np.testing.assert_array_equal(trabalho[serie.notna()], serie.dropna())
assert trabalho.notna().all() and tratada.notna().all()
assert not (suspeitos & ~diretos).any()
assert experimentos.groupby(['Repetição','Cenário']).size().eq(len(METODOS)).all()
assert avaliacoes.groupby('Etapa')['N avaliado'].nunique().eq(1).all()
print('Verificações finais concluídas: observados preservados, calendário mensal, máscaras, unidades e decomposição consistentes.')
print('Resultados exportados para:', SAIDA)
''')
md(r'''
### Limitações para mencionar no trabalho
- A base reúne boletins de diferentes datas; não é uma série histórica perfeitamente harmonizada por revisão.
- As estatísticas originais descrevem 108 valores disponíveis, dos quais nove são reconstituições contábeis.
- As 12 imputações são hipóteses, sobretudo os dez dezembros sem observações de referência. O total bimestral e os totais anuais fornecem informação adicional não imposta nesta análise.
- ADF/KPSS não resolvem sozinhos sazonalidade, quebras estruturais ou mudanças na variância; os resultados podem mudar com a imputação e a especificação.
- A detecção de outliers depende da decomposição, da janela e do limite escolhido. Suspeito estatístico não é erro comprovado.
- Tendência e ciclo são estimativas dependentes de filtros, com maior incerteza nas pontas.
- A escolha de modelos usa um ano de validação e um ano de teste com cobertura parcial. Uma avaliação mais forte usaria várias origens de previsão e dados revisados de forma consistente.

**Para explicar em sala:** “Primeiro preservei os dados oficiais. Depois apaguei valores conhecidos para comparar reposições, mantendo as lacunas originais separadas. Completei uma série de trabalho, testei sua estacionariedade, sinalizei desvios em relação à tendência e à sazonalidade e comparei tratamentos em uma simulação. Por fim, calculei componentes e médias móveis e comparei Holt/Holt-Winters em períodos futuros separados do treino.”
''')

def executar_e_salvar():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import IPython.display
    ns = {'__name__': '__main__'}
    for n, cel in enumerate([c for c in CELULAS if c['cell_type']=='code'], 1):
        outputs = []
        class Capture(io.StringIO):
            def write(self, s):
                if s:
                    if outputs and outputs[-1]['output_type']=='stream' and outputs[-1]['name']=='stdout':
                        outputs[-1]['text'] += s
                    else: outputs.append({'output_type':'stream','name':'stdout','text':s})
                return len(s)
        def display(obj):
            data = {'text/plain': repr(obj)}
            for method, mime in [('_repr_html_','text/html'),('_repr_markdown_','text/markdown')]:
                if hasattr(obj, method):
                    value = getattr(obj,method)()
                    if value is not None: data[mime] = value
            outputs.append({'output_type':'display_data','data':data,'metadata':{}})
        def show():
            for num in plt.get_fignums():
                fig = plt.figure(num); buf=io.BytesIO()
                fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
                outputs.append({'output_type':'display_data','data':{'image/png':base64.b64encode(buf.getvalue()).decode(), 'text/plain':'<Figure>'},'metadata':{}})
                plt.close(fig)
        IPython.display.display = display
        plt.show = show
        print(f'Executando célula {n}...', flush=True)
        with contextlib.redirect_stdout(Capture()):
            exec(compile(''.join(cel['source']), f'<celula {n}>', 'exec'), ns)
        cel['execution_count'] = n; cel['outputs'] = outputs
    nb = {'cells':CELULAS,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':sys.version.split()[0]}},'nbformat':4,'nbformat_minor':5}
    Path('energia_china_estudo_completo.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
    print(ns['texto'])
    print('Notebook salvo, com todas as células executadas.')

if __name__ == '__main__': executar_e_salvar()
