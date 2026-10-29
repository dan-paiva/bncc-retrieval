"""Rerank com LLM (Claude): reordena os candidatos da primeira etapa.

Decisões:
- **Saída estruturada com enum dinâmico**: o JSON Schema de cada chamada só aceita os códigos
  candidatos daquela consulta. O modelo não consegue inventar um código; mesmo assim a resposta
  é validada de novo aqui (defesa em profundidade).
- **O LLM só reordena**: quem decide o que entra no top-N é a busca barata. Se o modelo devolver
  menos de N códigos, o resto entra na ordem da primeira etapa, então o ranking nunca encolhe.
- **Cache em disco por hash do pedido**: rodar o benchmark de novo não gasta tokens, e quem
  clona o repositório reproduz os números sem chave (results/cache_llm.jsonl é versionado).
- **Custo medido**: tokens de entrada/saída de cada chamada vêm do `usage` da API.

A chave fica em ANTHROPIC_API_KEY (variável de ambiente ou arquivo .env, que está no .gitignore).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .bncc import COMPONENTES

CACHE_LLM = Path(__file__).resolve().parents[2] / "results" / "cache_llm.jsonl"
MODELO_PADRAO = "claude-opus-5-5"

# US$ por milhão de tokens (entrada, saída), tabela pública da Anthropic em out/2026.
PRECOS = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

SISTEMA = """Você é especialista em currículo escolar brasileiro e conhece a BNCC do Ensino Fundamental.
Recebe a descrição de uma aula escrita por um professor e uma lista de habilidades candidatas da BNCC.
Ordene as habilidades da mais para a menos provável de ser o foco principal da aula.
Julgue pelo que os alunos realmente fazem na aula, não por palavras em comum.
Responda só com o JSON pedido."""


def _schema(codigos: list[str], top: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "ranking": {
                "type": "array",
                "items": {"type": "string", "enum": codigos},
                "description": f"Até {top} códigos, do mais ao menos provável, sem repetir.",
            },
            "justificativa": {"type": "string", "description": "Uma frase sobre por que o 1º colocado."},
        },
        "required": ["ranking", "justificativa"],
        "additionalProperties": False,
    }


def _prompt(texto: str, componente: str, ano: int, candidatos: list[tuple[str, str]]) -> str:
    lista = "\n".join(f"- {c}: {d}" for c, d in candidatos)
    return (f"Componente: {COMPONENTES[componente]}\nAno: {ano}º\n\n"
            f"Descrição da aula:\n{texto}\n\nHabilidades candidatas:\n{lista}")


@dataclass
class Gasto:
    chamadas: int = 0
    do_cache: int = 0
    tokens_entrada: int = 0
    tokens_saida: int = 0
    por_modelo: dict[str, float] = field(default_factory=dict)

    @property
    def dolares(self) -> float:
        return sum(self.por_modelo.values())


class RerankLLM:
    def __init__(self, modelo: str = MODELO_PADRAO, esforco: str = "low", top: int = 10,
                 cliente=None, cache: Path = CACHE_LLM):
        self.modelo, self.esforco, self.top = modelo, esforco, top
        self._cliente = cliente
        self.cache_path = cache
        self.gasto = Gasto()
        self._cache = self._ler_cache()

    @property
    def nome(self) -> str:
        return f"llm:{self.modelo}"

    @property
    def cliente(self):
        if self._cliente is None:
            import anthropic
            from dotenv import load_dotenv

            load_dotenv()
            self._cliente = anthropic.Anthropic()
        return self._cliente

    def _ler_cache(self) -> dict[str, dict]:
        if not self.cache_path.exists():
            return {}
        with open(self.cache_path, encoding="utf-8") as f:
            return {r["chave"]: r for r in map(json.loads, f)}

    def _guardar(self, registro: dict) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        self._cache[registro["chave"]] = registro

    def _chamar(self, prompt: str, codigos: list[str]) -> dict:
        resposta = self.cliente.beta.messages.create(
            model=self.modelo,
            max_tokens=4000,
            system=SISTEMA,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.esforco,
                           "format": {"type": "json_schema", "schema": _schema(codigos, self.top)}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if resposta.stop_reason == "refusal":
            return {"ranking": [], "justificativa": "recusado", "usage": [0, 0], "modelo": resposta.model}
        texto = next(b.text for b in resposta.content if b.type == "text")
        dados = json.loads(texto)
        dados["usage"] = [resposta.usage.input_tokens, resposta.usage.output_tokens]
        dados["modelo"] = resposta.model
        return dados

    def reordenar(self, texto: str, componente: str, ano: int, candidatos: list[tuple[str, str]]) -> list[str]:
        codigos = [c for c, _ in candidatos]
        prompt = _prompt(texto, componente, ano, candidatos)
        chave = hashlib.sha256(f"{self.modelo}|{self.esforco}|{SISTEMA}|{prompt}".encode()).hexdigest()

        if chave in self._cache:
            registro = self._cache[chave]
            self.gasto.do_cache += 1
        else:
            registro = {"chave": chave, **self._chamar(prompt, codigos)}
            self._guardar(registro)
            self.gasto.chamadas += 1

        entrada, saida = registro["usage"]
        self.gasto.tokens_entrada += entrada
        self.gasto.tokens_saida += saida
        modelo = registro.get("modelo", self.modelo)  # pode ser o de fallback, se houve recusa
        pe, ps = PRECOS.get(modelo, PRECOS[MODELO_PADRAO])
        self.gasto.por_modelo[modelo] = self.gasto.por_modelo.get(modelo, 0.0) + (entrada * pe + saida * ps) / 1e6

        # Valida de novo: só códigos candidatos, sem repetição; completa com a ordem da 1ª etapa.
        validos = set(codigos)
        ranking = list(dict.fromkeys(c for c in registro["ranking"] if c in validos))
        return ranking + [c for c in codigos if c not in ranking]
