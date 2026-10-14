[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | **Português (BR)** | [Русский](README.ru.md)

# Quintet-Verify (QV)

**Inferência multiagente com restrições rígidas** · 9,17/10 em avaliação cega · ~US$0,36/execução · Python 3.11+

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
[![Stars](https://img.shields.io/github/stars/Qianjinqie/quintet-Verify?style=social)](https://github.com/Qianjinqie/quintet-Verify/stargazers)
![Cost](https://img.shields.io/badge/cost-~$0.36%2Frun-yellowgreen)
![BlindEval](https://img.shields.io/badge/Fable%205%20Blind%20Eval-9.17%2F10-brightgreen)

Um framework de controle cognitivo multiagente baseado em arquitetura cognitiva ancorada – usa uma divisão de trabalho estilo tribunal com cinco papéis (A/B/C/D/E) e traços cognitivos privados (✓/?/✗) para impor freios e contrapesos rígidos aos agentes no momento da inferência: sem fine-tuning, sem alteração de pesos, apenas restrições sobre “quem fala quando, com qual autoridade e com base em quais evidências.”

- Agnóstico de modelo · Python 3.11+ · LangGraph · Pydantic v2 · FastAPI

🌐 Demo online (implantada via workbuddy): https://a524b836ec13d5d9f.app.workbuddy.link

---

## Sumário

- [Resultados primeiro](#resultados-primeiro)
  - [Conclusão da avaliação cega do Fable 5](#conclusão-da-avaliação-cega-do-fable-5)
- [Restrições fundamentais](#restrições-fundamentais)
- [Por que não usar apenas um grande modelo?](#por-que-não-usar-apenas-um-grande-modelo)
- [Funcionalidades principais](#funcionalidades-principais)
- [Como ele é diferente de outros frameworks multiagente?](#como-ele-é-diferente-de-outros-frameworks-multiagente)
- [Explicação da qualidade](#explicação-da-qualidade)
- [Dados de execuções reais](#dados-de-execuções-reais)
- [Quem deve usá-lo?](#quem-deve-usá-lo)
- [Plugins](#plugins)
- [Início rápido](#início-rápido)
  - [Instalação](#instalação)
  - [Exemplo mínimo (LLM real)](#exemplo-mínimo-llm-real)
  - [Demo offline sem LLM](#demo-offline-sem-llm)
  - [Console web](#console-web)
- [Origens teóricas e declaração de independência](#origens-teóricas-e-declaração-de-independência)
  - [Inspirações](#inspirações)
  - [Diferenças arquiteturais fundamentais](#diferenças-arquiteturais-fundamentais)
  - [Resumo das diferenças principais](#resumo-das-diferenças-principais)
  - [Declaração de independência](#declaração-de-independência)

---

## Resultados primeiro

📊 **Conclusão da avaliação cega do Fable 5**

**Configuração da avaliação** – O Quintet-Verify roda no DeepSeek-V4-Flash-0731 (uma versão leve com 13 bilhões de parâmetros ativos; tarefas de documentos jurídicos usam o DeepSeek-V4-Pro-0813), enquanto o concorrente é o próprio modelo de avaliação (Fable 5), executado diretamente sem nenhum framework. O modelo de avaliação vê apenas dois rótulos anônimos, A e B — em outras palavras, sob a pontuação de um modelo forte, o modelo leve guiado pelo Quintet-Verify supera o F5.

A avaliação cega feita pelo Fable 5 atribuiu ao nosso projeto uma pontuação geral de 9,17/10, enquanto o concorrente (Fable 5) recebeu 8,0/10 – uma margem de +1,17.

As três tarefas foram vencidas:

- Revisão de código: 9,2 vs 8,5 (o concorrente F5 deixou passar problemas sutis como perda de mensagens no rpop, sobrescrita do campo de estado e vazamento previsível de task_id)  
- Avaliação de solução técnica: 9,5 vs 8,0 (o F5 calculou errado, tratando pairwise como pointwise na camada de ranking – um erro fatal)  
- Revisão de documento jurídico: 8,8 vs 7,5 (o F5 não apresentou um plano de verificação específico)

Nota: a tarefa de documento jurídico usou o DeepSeek-V4-Pro-0813; a revisão de código e a avaliação técnica usaram o DeepSeek-V4-Flash-0731. O framework rodou em modo padrão – intensidade alta.

O avaliador (F5) concluiu qualitativamente:

> O sistema avaliado supera o sujeito em cinco dimensões: profundidade técnica, precisão computacional, identificação de riscos, suporte de dados e rigor profissional. Adequado para cenários de alto risco, como decisões de investimento, due diligence técnica, revisões de implantação em pré-produção e auditorias de conformidade.

Nota: na prática, o QV faz muitas chamadas de API, incorrendo em custos mais altos – ele troca dinheiro e tempo por qualidade.

---

## Restrições fundamentais

- A não escreve o texto principal, B não altera o framework, C não fornece revisões do texto completo, D não redige em nome dos outros, E não derruba vereditos diretamente.  
- Os marcadores cognitivos ✓/?/✗ existem como traços privados e nunca entram nas decisões de roteamento.  
- Devoluções devem incluir coordenadas de âncora; o retrabalho fica limitado às regiões designadas (declarações de ripple restringem o escopo das alterações).

---

## Por que não usar apenas um grande modelo?

A premissa de um modelo único de que “gerar já é definitivo” é a causa raiz de alucinações chegarem diretamente ao ponto final: ele gera, revisa e confirma por si mesmo – o supervisor e o supervisionado compartilham o mesmo ponto de vista. O Quintet-Verify quebra esse circuito fechado em cinco papéis com permissões mutuamente travadas –

Os freios e contrapesos evoluem de “autodisciplina” para “disciplina externa”: o supervisor e o supervisionado não compartilham um ponto de vista; toda devolução é confinada dentro das coordenadas de âncora, de modo que o retrabalho nunca provoca desvios no texto inteiro.

---

## Funcionalidades principais

- **Arquitetura cognitiva de canal duplo**: o canal público (visível ao roteamento) e o Dense Track privado (✓/?/✗) são fisicamente isolados, com três camadas de proteção (asserções estruturais / de tipo / de runtime) garantindo que a cognição privada nunca entre no roteamento.  
- **Cinco papéis com freios e contrapesos rígidos**: A (Arquiteto) · B (Executor) · C (Verificador) · D (Juiz) · E (Jurado) – cada um com permissões travadas e zonas proibidas.  
- **Três regras de ferro (aplicadas em código)**: ① isolamento de confiança ② modificação incremental (coordenadas de âncora + declarações de ripple) ③ ordem de silêncio com circuit breaker (max_iterations: 1–10, padrão 5; D pode emitir uma ordem de silêncio).  
- **Níveis de intensidade**: low / high (padrão) / max; no modo estrito, os defeitos devem vir acompanhados de uma cadeia de impacto, e D não pode aprovar se a confiança for insuficiente.  
- **Modo rápido**: apenas o triângulo B/C/D, sem esboço, sem ramificação de apelação – adequado para tarefas simples e cenários de baixo custo.  
- **Governança de overfitting**: C não tem cota obrigatória (zero defeitos é permitido, mas deve incluir uma declaração de cobertura); D pode considerar C excessivamente pedante e devolvê-lo para revisão (no máximo 2 vezes).  
- **Mecanismo de plugins**: registro PLUGINS + flags (todas desativadas por padrão); o opcional qv_extras.py pode ser montado via pre/post_act_hooks padronizados – veja [Plugins](#plugins).  
- **Workflow personalizável (v1.4)**: edição front-end das definições de workflow, criação de novos papéis herdando as permissões AE, validação estática e configuração de LLM independente por papel.

---

## Como ele é diferente de outros frameworks multiagente?

- **AutoGen / CrewAI / MetaGPT**: todos suportam colaboração multiagente e são agnósticos de modelo, mas carecem de três capacidades: freios e contrapesos rígidos no momento da inferência, auditoria de traços cognitivos privados e modificação incremental (economia de tokens).  
- **Quintet-Verify**: além de suportar colaboração multiagente e agnosticismo de modelo, oferece adicionalmente:  
  - Freios e contrapesos rígidos no momento da inferência (intertravamento de permissões estilo tribunal)  
  - Traços cognitivos privados auditáveis (Dense Track)  
  - Modificação incremental (coordenadas de âncora + declarações de ripple, economizando em média ~80% do contexto)

A maioria dos frameworks faz “deixar a IA colaborar”; o Quintet-Verify faz “deixar a IA impor freios e contrapesos entre si” – o primeiro busca eficiência de saída, o segundo busca confiabilidade das conclusões.

---

## Explicação da qualidade

Em comparação com usar um único modelo diretamente, os ganhos do Quintet-Verify vêm de freios e contrapesos estruturais, e não de quantidades maiores de parâmetros:

- **Comparado com um modelo puro**: as alucinações não chegam mais diretamente ao ponto final – precisam primeiro passar pela revisão de C e pelo veredito de D, e cada devolução é confinada dentro das coordenadas de âncora. A conclusão de zero defeitos de C e o julgamento de overfitting de D evitam simultaneamente o atrito interno de “criticar por criticar”.  
- **Comparado com o J-Space puro**: a gestão cognitiva densa do J-Space opera dentro de um único modelo; o teto da autossupervisão é “se pegar a si mesmo”. O Quintet-Verify mantém ✓/?/✗ como traços cognitivos privados enquanto delega a autoridade de veredito a papéis independentes – elevando os freios e contrapesos da autodisciplina à disciplina externa.

O exposto é uma comparação qualitativa no nível de mecanismo; a eficácia prática é em geral muito superior à do seu modelo de base – veja Resultados primeiro para detalhes.

---

## Dados de execuções reais

Execução real em modo completo (agosto de 2026, DeepSeek-V4-Pro-0813, tarefa de revisão de documento jurídico):

- Consumo de tokens: ~130.000  
- Chamadas de API: 19  
- Custo: ~¥2,3 (~US$0,36)  
- Veredito final: D aprovou com confiança de 90/100

A saída concluiu “condicionalmente viável com arquitetura híbrida”: asserções empíricas sem suporte de dados no rascunho inicial foram ou removidas por completo ou rebaixadas a hipóteses pendentes de verificação após duas devoluções; os critérios de aceitação foram reescritos em termos auditáveis.

---

## Quem deve usá-lo?

- Desenvolvedores que constroem aplicações RAG mas são atormentados por alucinações  
- Cenários jurídicos, financeiros e de saúde que exigem decisões de IA auditáveis  
- Pesquisadores que querem experimentar freios e contrapesos multiagente sem escrever orquestrações complexas  
- Equipes satisfeitas com a eficiência de colaboração do AutoGen/CrewAI, mas que exigem maior confiabilidade das conclusões  
- Programadores que usam IA para escrever código de alta qualidade  
- Pesquisadores que usam IA como assistência para pesquisa matemática

---

## Plugins

- **Pacote de plugins**: o opcional `qv_extras.py` é distribuído separadamente em [Quintet-Verify-plugin](https://github.com/Qianjinqie/Quintet-Verify-plugin) – coloque-o ao lado de `quintet_verify.py` e ele será montado automaticamente na importação (sem necessidade de configuração).  
- **Sem o pacote de plugins**: o workflow central roda sem alterações; as 7 flags de plugins simplesmente permanecem desativadas.

---

## Início rápido

### Instalação

```bash
git clone https://github.com/Qianjinqie/quintet-Verify.git
cd quintet-Verify
pip install -r requirements.txt
```

### Exemplo mínimo (LLM real)

```python
from quintet_verify import (
    DEFAULT_FLAGS,
    LLMConfig,
    PublicState,
    QuintetState,
    build_graph,
    build_llm_agents,
)

# 1. Configurar o modelo (endpoint compatível com OpenAI; compartilhado ou por papel)
cfg = LLMConfig(
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
)
agents = build_llm_agents(cfg)  # retorna os cinco papéis (A, B, C, D, E)

# 2. Construir o grafo e executar (interrupt_before_e=False significa não pausar antes de E para feedback do usuário)
graph = build_graph(*agents, interrupt_before_e=False)
final = graph.invoke(
    QuintetState(
        public=PublicState(
            task="Evaluate the feasibility of a technical solution and provide a conclusion.",
            user_satisfied=True,   # satisfeito → E concorda com o veredito, o workflow termina
            intensity="high",      # níveis de intensidade: low / high / max
            flags=dict(DEFAULT_FLAGS),  # interruptores de plugins, todos desativados por padrão
        )
    )
)

# 3. Ler os resultados
pub = final["public"]
print(pub.verdict.approved, pub.verdict.confidence_score)  # veredito e confiança de D
for section in pub.draft.sections:
    print(section.anchor_id, section.content)
```

### Demo offline sem LLM

```python
from quintet_verify import PublicState, QuintetState, build_graph, build_stub_agents

final = build_graph(*build_stub_agents(), interrupt_before_e=False).invoke(
    QuintetState(public=PublicState(task="Demo task", user_satisfied=True))
)
print(final["public"].verdict)
```

### Console web

```bash
python server.py  # porta padrão 8000, abra no navegador
```

O console suporta: pipelines Standard / Fast / Custom, níveis de intensidade, alternância de plugins, configuração de API por papel, edição das definições de workflow com validação em tempo real, sobrescrita de prompts por papel e visualização de execuções (fluxo de eventos / rascunho / Dense Track).

---

## Origens teóricas e declaração de independência

### Inspirações

O Quintet-Verify se inspira em trabalhos anteriores nos seguintes aspectos:

· Os símbolos de marcadores cognitivos (✓/?/✗) e o nome “traços cognitivos privados” (Dense Track): derivados do J-Space Cognition Suite V3.6 (doi:10.5281/zenodo.21971181). Esse trabalho foi o primeiro a propor sistematicamente o uso de símbolos compactos dentro de um único modelo para gerenciar estados cognitivos, e validou o efeito positivo dos “marcadores introspectivos” sobre a qualidade do raciocínio.
· A forma organizacional dos freios e contrapesos estilo tribunal com múltiplos papéis: inspirada no sistema de debate multimodelo Quintet Verification apresentado em b23.tv/z3C0vVM, que demonstrou a viabilidade de “transformar alucinações de geração em jogos de consenso por meio da separação de papéis e de procedimentos de objeção”.

Expressamos nosso respeito e gratidão pelas contribuições originais dos trabalhos acima.

### Diferenças arquiteturais fundamentais

Embora o Quintet-Verify tome emprestada a nomenclatura de símbolos do J-Space e a forma organizacional do Quintet Verification, sua arquitetura cognitiva subjacente difere de forma irredutível de ambos. Essas diferenças não são melhorias incrementais, mas sim uma redefinição de três premissas fundamentais: sujeito cognitivo, topologia de informação e natureza do controle.

| Aspecto | J-Space (Teoria do Espaço de Trabalho Global) | Quintet Verification (debate multimodelo) | Quintet-Verify (Teoria Cognitiva Ancorada) |
|---|---|---|---|
| Sujeito cognitivo | Dentro de um único modelo | Múltiplos modelos independentes (sem estado) | Múltiplos agentes independentes (cada um com estado privado) |
| Suporte do estado cognitivo | Janela de contexto global | Saídas independentes de cada modelo | Estado compartilhado público + traços privados (fisicamente isolados) |
| Fluxo de informação | Broadcast global + acesso competitivo | Transmissão completa estilo debate | Referências ancoradas (via anchor_id + checksum para fatos compartilhados) |
| Mecanismo de modificação | Reescrita completa (depende de correção introspectiva) | Reescrita completa (depende de correção por debate) | Modificação incremental (coordenadas precisas + declarações de ripple) |
| Natureza do controle | Restrições flexíveis guiadas por prompt | Restrições flexíveis guiadas por prompt | Validação em tempo de compilação + guardas de roteamento (restrições rígidas) |
| Mecanismo de recuperação | Rollback de registro textual | Sem recuperação padronizada | Checkpoints atômicos de estado (rollback exato para qualquer rodada) |
| Freios e contrapesos | Autossupervisão (mesmo ponto de vista) | Debate entre modelos (pontos de vista independentes) | Intertravamento de permissões + procedimento de tribunal (separação de pontos de vista + rigidez procedimental) |
| Auditabilidade | Texto de cadeia de raciocínio (rollback não estrutural) | Texto de debate (rollback não estrutural) | Auditoria estruturada do Dense Track (evidências de verificação rastreáveis para ✓ e evidências de falsificação para ✗) |

### Resumo das diferenças principais

· O J-Space trata de “como uma única pessoa pensa melhor” (introspecção profunda de uma única consciência);
· O Quintet Verification trata de “como um grupo alcança consenso por meio do debate” (verificação externa via múltiplos pontos de vista);
· O Quintet-Verify trata de “como um grupo de agentes com estados cognitivos independentes converge para conclusões verificáveis por meio de regras rígidas” (governança constitucional da cognição distribuída).

A contribuição única do Quintet-Verify é a primeira combinação de “traços cognitivos privados (Dense Track)” com uma “matriz de permissões rígida”, permitindo que sistemas multiagente alcancem uma confiabilidade em nível de sistema auditável, rastreável e encerrável à força, mantendo a profundidade do raciocínio individual.

### Declaração de independência

Com base nas diferenças arquiteturais fundamentais acima, o Quintet-Verify declara:

· Não é uma obra derivada: o Quintet-Verify não é uma obra derivada nem um fork do J-Space Cognition Suite V3.6. Os dois adotam implementações de código, topologias cognitivas e filosofias de controle completamente diferentes.
· Propriedade intelectual independente: todo o código do Quintet-Verify foi escrito de forma independente pela nossa equipe; nenhum código-fonte, template de prompt ou texto de protocolo do J-Space ou do Quintet Verification foi copiado ou traduzido.
· Citação, não dependência: este sistema lista o J-Space e o Quintet Verification como “referências teóricas”, e não como “dependências técnicas”. A remoção dessas referências não afeta a operação completa do workflow central do Quintet-Verify.
· Integridade acadêmica: esta declaração visa rastrear com precisão as origens intelectuais, definindo claramente os limites das nossas contribuições originais. Encorajamos os leitores a realizar estudos comparativos entre os trabalhos acima e este sistema para compreender os diferentes caminhos evolutivos das arquiteturas cognitivas distribuídas.

Recomendação para os leitores:

· Se você tem interesse em “como um único modelo melhora o raciocínio por meio de marcadores introspectivos”, consulte o J-Space Cognition Suite V3.6.
· Se você tem interesse em “como múltiplos modelos alcançam consenso por meio do debate”, consulte o sistema Quintet Verification.
· Se você tem interesse em “como fazer a colaboração multiagente produzir conclusões auditáveis e verificadas por meio de regras rígidas”, continue usando o Quintet-Verify.
