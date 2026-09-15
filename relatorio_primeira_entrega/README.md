# Relatório — Atividade I de Econometria II

O arquivo principal é `relatorio.tex`. Ele utiliza as pastas `tabelas/` e `figuras/`, que devem permanecer junto ao arquivo ao compilar ou enviar ao Overleaf.

O PDF compilado está em `relatorio.pdf` (8 páginas). A compilação foi conferida com Tectonic, incluindo referências e disposição de tabelas e figuras.

O relatório cobre os itens I a V solicitados e não inclui testes de estacionariedade. Os números são provenientes do notebook `energia_china_estudo_completo.ipynb` e dos CSVs associados. As figuras foram extraídas das saídas incorporadas ao notebook.

O EQM médio da comparação de imputação foi calculado a partir do quadrado do RMSE de cada repetição, com média por cenário e depois entre cenários. Não se utilizou o quadrado do RMSE médio.

## Compilação

No Overleaf, envie todo o conteúdo desta pasta e selecione `relatorio.tex` como documento principal. Utilize pdfLaTeX.

Localmente, dentro desta pasta:

```bash
pdflatex -interaction=nonstopmode -halt-on-error relatorio.tex
pdflatex -interaction=nonstopmode -halt-on-error relatorio.tex
```

A segunda execução resolve referências a tabelas, figuras e bibliografia. Também é possível compilar com `tectonic relatorio.tex`.
