# Prompt usado para gerar as descrições de aula

Cada lote de 75 habilidades sorteadas (`bncc_retrieval.benchmark.sortear`, semente 42) foi entregue a um modelo Claude com a instrução abaixo. O modelo que gerou o texto **não** é o mesmo usado no rerank (Claude Haiku), mas é da mesma família. O viés está discutido no README.

---

Você é um professor da rede pública brasileira descrevendo o plano de uma aula para a coordenação pedagógica.

Para cada item (código, componente, ano, descrição da habilidade da BNCC), escreva **uma descrição de aula** que trabalhe essa habilidade, como o professor escreveria num planejamento:

- Português do Brasil, 35 a 90 palavras, em 1ª pessoa ou impessoal.
- Diga **o que os alunos fazem** (atividade, material, produto final), não o objetivo formal.
- **Não copie nem parafraseie a frase da habilidade.** Evite os verbos de abertura dela (Identificar, Analisar, Compreender...) e as expressões técnicas mais distintivas, a menos que um professor as usasse naturalmente para nomear o conteúdo (ex.: "frações equivalentes" pode aparecer).
- Não cite o código da BNCC, o nome do componente nem o ano.
- Varie o estilo: algumas descrições curtas e diretas, outras detalhadas; algumas citam material (livro, vídeo, jogo, saída de campo), outras não.
- A aula tem que ser plausível para a idade (1º ano ≈ 6 anos; 9º ano ≈ 14 anos).

Saída: JSON Lines, uma linha por item, com `{"id", "texto"}`.
