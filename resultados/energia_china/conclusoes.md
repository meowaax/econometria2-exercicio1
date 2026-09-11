**Imputação:** Harmônica log teve o menor MAE médio nos cenários avaliados: **21.92 TWh**. Isso não valida automaticamente dezembro, que nunca foi observado nesta coleta.

**Estacionariedade:** no log em nível, a leitura conjunta foi: **Compatível com não estacionariedade**. Compare a tabela de diferenças e a alternativa linear; os testes não incorporam toda a incerteza da imputação.

**Outliers:** foram sinalizados **6 valores diretamente publicados** como suspeitos, preservados na base oficial e winsorizados apenas em uma cópia de estudo. No experimento artificial, o detector encontrou **6 de 6** alterações. O menor MAE nas posições injetadas foi de **Reconstrução STL**; confira também o erro nos demais pontos.

**Componentes:** a média móvel torna o movimento geral mais visível. A STL separa sazonalidade e resíduo; o HP divide a componente suave em tendência e ciclo estimados. A soma dos quatro componentes foi conferida numericamente. O ciclo é dependente do filtro, não uma prova de recessões ou expansões.

**Previsão:** a escolha pelo MAE de 2023 foi **Holt-Winters aditivo** (25.09 TWh). Mantida essa escolha, o MAE nos **9 meses diretamente observados de 2024** foi **18.84 TWh**. Esse número não representa o erro nos 12 meses de 2024.
