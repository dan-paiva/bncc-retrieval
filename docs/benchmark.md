# Como o benchmark foi construído

## Corpus
As 1301 habilidades do Ensino Fundamental, extraídas do PDF oficial da BNCC homologada (MEC, dez/2017), com `python -m bncc_retrieval.extract`. O PDF usado tem SHA-256 `7d0004aacde591d4d898cbdf950ab7ed28e875f837dab12bb88cb15567cdf180`. Cada linha de `data/bncc_ef.csv` traz código, componente, anos cobertos e descrição.

## Consultas
Não existe um conjunto público de aulas rotuladas com códigos da BNCC, então criei um:

1. **Sorteio** (`benchmark.sortear`, semente 42): 300 habilidades, estratificadas por componente na proporção do corpus, com no mínimo 10 por componente. Para habilidades de bloco (ex.: EF35LP01), sorteia-se um ano concreto da aula dentro do bloco.
2. **Escrita**: um modelo Claude recebeu cada habilidade e escreveu uma descrição de aula como um professor faria, com a instrução explícita de **não parafrasear** a habilidade (prompt completo em [`prompt_geracao.md`](prompt_geracao.md)).
3. **Checagem de vazamento**: nenhuma aula cita o código. A sobreposição de palavras (Jaccard sobre tokens) entre a aula e a habilidade correta tem mediana de **0,07** e máximo de 0,21: a busca não pode depender de copiar texto.
4. **Revisão humana**: amostra de 30 aulas em [`revisao_humana.md`](revisao_humana.md).
5. **Divisão**: 100 consultas em `dev` (para ajustar) e 200 em `test` (só para o número final).

## Métricas
- **MRR**: média de 1/posição da habilidade correta.
- **Hit@k**: fração das aulas em que a correta aparece entre as k primeiras. Hit@3 é a métrica de produto: o professor confirma uma de três sugestões.
- **nDCG@10**.
- **IC 95%** por bootstrap (10 mil reamostragens) e **bootstrap pareado** para comparar métodos nas mesmas consultas.

## Limitações (leia antes de citar os números)
- **Dado sintético.** Aulas reais são mais bagunçadas: misturam várias habilidades, têm erros de digitação e contexto local. Os números são um limite superior otimista.
- **Um rótulo por aula.** Uma aula pode trabalhar habilidades vizinhas (ex.: EF15LP e EF35LP se sobrepõem), e o benchmark conta isso como erro. O desempenho real tende a ser melhor do que o medido aqui, sobretudo em Língua Portuguesa.
- **Mesma família de modelo.** As aulas foram escritas por um modelo Claude, e o rerank também usa Claude. Isso pode favorecer o rerank com LLM; os métodos locais não têm esse viés.
