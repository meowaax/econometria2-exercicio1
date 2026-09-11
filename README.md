# Consumo de eletricidade na China

Trabalho de Econometria II sobre a série mensal de consumo de eletricidade na China entre 2015 e 2024, organizada a partir de publicações da [National Energy Administration (NEA)](https://www.nea.gov.cn/sjzz/ghs/zjgx.htm).

A análise inclui estatística descritiva, testes de estacionariedade (ADF e KPSS), comparação de métodos de imputação, identificação e tratamento de outliers, médias móveis, decomposição em tendência, sazonalidade e ciclo, além de suavização de Holt e Holt-Winters.

## Arquivos principais

- [energia_china_estudo_completo.ipynb](energia_china_estudo_completo.ipynb): notebook principal, com explicações, cálculos, tabelas e gráficos já executados.
- [energia_china_analise_descritiva.ipynb](energia_china_analise_descritiva.ipynb): análise inicial e documentação da fonte.
- `dados/energia_china/`: dados coletados, páginas de referência e procedência dos valores.
- `resultados/energia_china/`: tabelas, gráficos e conclusões do estudo completo.
- `coletar_energia_china.py`: coleta os boletins da NEA.
- `preparar_energia_china.py`: organiza os dados das páginas arquivadas.
- `gerar_notebook_estudo_china.py`: gera e executa o notebook completo.

## Como executar

Na pasta do projeto, crie um ambiente Python e instale as dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy statsmodels matplotlib ipython ipykernel jupyterlab
```

Abra `energia_china_estudo_completo.ipynb` no VS Code ou no JupyterLab, selecione o ambiente criado e execute as células em ordem. Para iniciar o JupyterLab:

```bash
jupyter lab
```

Também é possível gerar novamente o notebook e seus resultados pelo terminal:

```bash
python gerar_notebook_estudo_china.py
```

Esse comando sobrescreve o notebook completo e os resultados gerados. A análise usa os dados locais; somente uma nova coleta exige internet e o comando `curl`.

## Observações sobre os dados

A grade contém 120 meses: 99 valores publicados diretamente, 9 janeiros reconstituídos a partir do mesmo boletim e 12 lacunas na coleta. O consumo é expresso em **TWh**.

Os dados oficiais são preservados. Preenchimentos e tratamentos são identificados como estimativas ou alterações didáticas. Um valor estatisticamente atípico não é necessariamente um erro de registro; as limitações de cada método estão explicadas no notebook.
