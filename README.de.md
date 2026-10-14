[English](README.en.md) | [简体中文](README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md) | **Deutsch** | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)

# Quintet-Verify (QV)

**Hart eingeschränkte Multi-Agenten-Inferenz** · 9.17/10 Blindbewertung · ~$0.36/Lauf · Python 3.11+

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
[![Stars](https://img.shields.io/github/stars/Qianjinqie/quintet-Verify?style=social)](https://github.com/Qianjinqie/quintet-Verify/stargazers)
![Cost](https://img.shields.io/badge/cost-~$0.36%2Frun-yellowgreen)
![BlindEval](https://img.shields.io/badge/Fable%205%20Blind%20Eval-9.17%2F10-brightgreen)

Ein Multi-Agenten-Framework zur kognitiven Steuerung auf Basis einer verankerten kognitiven Architektur – mit einer Aufgabenteilung im Tribunal-Stil über fünf Rollen (A/B/C/D/E) und privaten kognitiven Spuren (✓/?/✗), die den Agenten zur Inferenzzeit starre Checks and Balances auferlegt: kein Fine-Tuning, keine Gewichtsänderungen, nur Beschränkungen darüber, „wer wann spricht, mit welcher Autorität und auf Basis welcher Evidenz.“

- Modellagnostisch · Python 3.11+ · LangGraph · Pydantic v2 · FastAPI

🌐 Online-Demo (bereitgestellt über workbuddy): https://a524b836ec13d5d9f.app.workbuddy.link

---

## Inhaltsverzeichnis

- [Ergebnisse zuerst](#ergebnisse-zuerst)
  - [Fazit der Blindbewertung durch Fable 5](#fazit-der-blindbewertung-durch-fable-5)
- [Zentrale Beschränkungen](#zentrale-beschränkungen)
- [Warum nicht einfach ein großes Modell verwenden?](#warum-nicht-einfach-ein-großes-modell-verwenden)
- [Kernfunktionen](#kernfunktionen)
- [Wie unterscheidet es sich von anderen Multi-Agent-Frameworks?](#wie-unterscheidet-es-sich-von-anderen-multi-agent-frameworks)
- [Erklärung der Qualität](#erklärung-der-qualität)
- [Daten aus realen Läufen](#daten-aus-realen-läufen)
- [Für wen ist es gedacht?](#für-wen-ist-es-gedacht)
- [Plugins](#plugins)
- [Schnellstart](#schnellstart)
  - [Installation](#installation)
  - [Minimales Beispiel (echtes LLM)](#minimales-beispiel-echtes-llm)
  - [Offline-Demo ohne LLM](#offline-demo-ohne-llm)
  - [Web-Konsole](#web-konsole)
- [Theoretische Ursprünge und Unabhängigkeitserklärung](#theoretische-ursprünge-und-unabhängigkeitserklärung)
  - [Inspirationen](#inspirationen)
  - [Grundlegende architektonische Unterschiede](#grundlegende-architektonische-unterschiede)
  - [Zusammenfassung der Kernunterschiede](#zusammenfassung-der-kernunterschiede)
  - [Unabhängigkeitserklärung](#unabhängigkeitserklärung)

---

## Ergebnisse zuerst

📊 **Fazit der Blindbewertung durch Fable 5**

**Aufbau der Bewertung** – Quintet-Verify läuft auf DeepSeek-V4-Flash-0731 (einer leichten Version mit 13 Milliarden aktiven Parametern; Aufgaben mit Rechtsdokumenten verwenden DeepSeek-V4-Pro-0813), während der Konkurrent das Bewertungsmodell selbst ist (Fable 5), das ohne jegliches Framework direkt läuft. Das Bewertungsmodell sieht nur zwei anonyme Labels, A und B – das heißt: Bei Bewertung durch ein starkes Modell übertrifft das von Quintet-Verify angetriebene leichte Modell das F5.

Die Blindbewertung durch Fable 5 vergab für unser Projekt eine Gesamtwertung von 9.17/10, während der Konkurrent (Fable 5) 8.0/10 erhielt – ein Abstand von +1.17.

Alle drei Aufgaben wurden gewonnen:

- Code Review: 9.2 vs 8.5 (Konkurrent F5 übersah subtile Probleme wie Nachrichtenverlust bei rpop, das Überschreiben des State-Felds und die vorhersagbare Preisgabe von task_id)  
- Bewertung technischer Lösungen: 9.5 vs 8.0 (F5 berechnete in der Ranking-Schicht pairwise fälschlich als pointwise – ein fataler Fehler)  
- Prüfung von Rechtsdokumenten: 8.8 vs 7.5 (F5 lieferte keinen konkreten Verifizierungsplan)

Hinweis: Die Aufgabe mit Rechtsdokumenten verwendete DeepSeek-V4-Pro-0813; Code Review und Bewertung technischer Lösungen verwendeten DeepSeek-V4-Flash-0731. Das Framework lief im Standardmodus – hohe Intensität.

Der Bewerter (F5) kam qualitativ zu folgendem Schluss:

> Das bewertete System übertrifft den Vergleichsgegenstand in fünf Dimensionen: technische Tiefe, Rechengenauigkeit, Risikoerkennung, Datenuntermauerung und professionelle Strenge. Geeignet für Szenarien mit hohen Einsätzen wie Investitionsentscheidungen, technische Due Diligence, Reviews vor Produktionsbereitstellungen und Compliance-Audits.

Hinweis: In der Praxis führt QV viele API-Aufrufe durch und verursacht dadurch höhere Kosten – es tauscht Geld und Zeit gegen Qualität.

---

## Zentrale Beschränkungen

- A schreibt den Haupttext nicht, B verändert das Framework nicht, C liefert keine Revisionen des Gesamttexts, D schreibt nicht stellvertretend für andere, E kippt Entscheidungen nicht direkt.  
- Kognitive Marker ✓/?/✗ existieren als private Spuren und fließen nie in Routing-Entscheidungen ein.  
- Zurückverweisungen müssen Ankerkoordinaten enthalten; Nacharbeiten sind auf ausgewiesene Bereiche beschränkt (Ripple-Deklarationen begrenzen den Umfang der Änderungen).

---

## Warum nicht einfach ein großes Modell verwenden?

Das Prinzip „Generieren bedeutet Finalität“ eines einzelnen Modells ist die Grundursache dafür, dass Halluzinationen direkt bis zum Endpunkt gelangen: Es generiert, prüft und bestätigt sich selbst – Überwachende und Überwachte teilen sich denselben Standpunkt. Quintet-Verify zerbricht diese geschlossene Schleife in fünf Rollen mit gegenseitig verriegelten Berechtigungen –

Checks and Balances entwickeln sich von „Selbstdisziplin“ zu „Fremddisziplin“: Überwachende und Überwachte teilen sich keinen Standpunkt; jeder Zurückverweis ist auf Ankerkoordinaten beschränkt, sodass Nacharbeiten nie dazu führen, dass der gesamte Text abdriftet.

---

## Kernfunktionen

- **Zweikanalige kognitive Architektur**: Der öffentliche (für das Routing sichtbare) und der private Dense Track (✓/?/✗) sind physisch isoliert; drei Schutzebenen (strukturell / typbezogen / Laufzeit-Assertions) stellen sicher, dass private Kognition niemals ins Routing gelangt.  
- **Starre Checks and Balances über fünf Rollen**: A (Architect) · B (Executor) · C (Verifier) · D (Judge) · E (Juror) – jede mit verriegelten Berechtigungen und verbotenen Zonen.  
- **Drei eiserne Regeln (im Code erzwungen)**: ① Konfidenz-Isolation ② inkrementelle Modifikation (Ankerkoordinaten + Ripple-Deklarationen) ③ Circuit-Breaker-Sprechverbot (max_iterations: 1–10, Standard 5; D kann ein Sprechverbot verhängen).  
- **Intensitätsstufen**: low / high (Standard) / max; im strengen Modus müssen Mängel mit einer Auswirkungskette begleitet werden, und D darf bei unzureichender Konfidenz nicht durchlassen.  
- **Schnellmodus**: nur das Dreieck B/C/D, keine Gliederung, kein Berufungszweig – geeignet für einfache Aufgaben und kostengünstige Szenarien.  
- **Overfitting-Governance**: C hat keine Pflichtquote (null Mängel sind erlaubt, muss aber eine Abdeckungserklärung enthalten); D kann C als übertrieben pedantisch einstufen und es zur Nachprüfung zurückgeben (maximal 2-mal).  
- **Plugin-Mechanismus**: PLUGINS-Registry + Flags (alle standardmäßig aus); das optionale qv_extras.py lässt sich über standardisierte pre/post_act_hooks einbinden – siehe [Plugins](#plugins).  
- **Anpassbarer Workflow (v1.4)**: Bearbeitung von Workflow-Definitionen im Front-End, Erstellen neuer Rollen mit Vererbung der AE-Berechtigungen, statische Validierung und unabhängige LLM-Konfiguration pro Rolle.

---

## Wie unterscheidet es sich von anderen Multi-Agent-Frameworks?

- **AutoGen / CrewAI / MetaGPT**: alle unterstützen Multi-Agenten-Kollaboration und sind modellagnostisch, ihnen fehlen jedoch drei Fähigkeiten: starre Checks and Balances zur Inferenzzeit, Auditing privater kognitiver Spuren und inkrementelle Modifikation (Token-Ersparnis).  
- **Quintet-Verify**: zusätzlich zur Unterstützung von Multi-Agenten-Kollaboration und Modellagnostik bietet es:  
  - Starre Checks and Balances zur Inferenzzeit (tribunal-artige Berechtigungsverschränkung)  
  - Audierbare private kognitive Spuren (Dense Track)  
  - Inkrementelle Modifikation (Ankerkoordinaten + Ripple-Deklarationen, spart im Durchschnitt ~80 % Kontext)

Die meisten Frameworks tun „Lass KI kollaborieren“; Quintet-Verify tut „Lass KI sich gegenseitig kontrollieren“ – Ersteres verfolgt Ausgabeneffizienz, Letzteres die Verlässlichkeit der Schlussfolgerungen.

---

## Erklärung der Qualität

Im Vergleich zur direkten Verwendung eines einzelnen Modells stammen die Gewinne von Quintet-Verify aus strukturellen Checks and Balances, nicht aus einer größeren Parameterzahl:

- **Im Vergleich zu einem reinen Modell**: Halluzinationen gelangen nicht mehr direkt zum Endpunkt – sie müssen zunächst die Prüfung durch C und die Entscheidung durch D passieren, und jeder Zurückverweis ist auf Ankerkoordinaten beschränkt. Das Null-Mängel-Fazit von C und das Overfitting-Urteil von D verhindern zugleich interne Reibung durch „Mäkeln zum Selbstzweck“.  
- **Im Vergleich zu reinem J-Space**: Das dichte kognitive Management von J-Space arbeitet innerhalb eines einzelnen Modells; die Obergrenze der Selbstüberwachung besteht darin, sich selbst zu ertappen. Quintet-Verify behält ✓/?/✗ als private kognitive Spuren bei, delegiert die Entscheidungsgewalt jedoch an unabhängige Rollen – damit werden Checks and Balances von Selbstdisziplin zu Fremddisziplin weiterentwickelt.

Das Obige ist ein qualitativer Vergleich auf Mechanismusebene; in der Praxis ist die Wirksamkeit im Allgemeinen weit überlegen gegenüber dem Basismodell – Einzelheiten siehe Ergebnisse zuerst.

---

## Daten aus realen Läufen

Echter Lauf im vollständigen Modus (August 2026, DeepSeek-V4-Pro-0813, Aufgabe: Prüfung von Rechtsdokumenten):

- Tokenverbrauch: ~130,000  
- API-Aufrufe: 19  
- Kosten: ~¥2.3 (~$0.36)  
- Letzte Entscheidung: D genehmigt mit Konfidenz 90/100

Die Ausgabe kam zum Ergebnis „bedingt machbar mit hybrider Architektur“: empirische Behauptungen ohne Datenuntermauerung im Erstentwurf wurden nach zwei Zurückverweisen entweder vollständig entfernt oder zu noch zu verifizierenden Hypothesen herabgestuft; die Abnahmekriterien wurden in audierbare Formulierungen umgeschrieben.

---

## Für wen ist es gedacht?

- Entwickler, die RAG-Anwendungen bauen, aber von Halluzinationen geplagt werden  
- Rechts-, Finanz- und Gesundheitsszenarien, die audierbare KI-Entscheidungen erfordern  
- Forschende, die Multi-Agenten-Checks and Balances erleben wollen, ohne komplexe Orchestrierungen zu schreiben  
- Teams, die mit der Kollaborationseffizienz von AutoGen/CrewAI zufrieden sind, aber eine höhere Verlässlichkeit der Schlussfolgerungen verlangen  
- Programmierer, die mit KI hochwertigen Code schreiben  
- Forschende, die KI als Unterstützung für mathematische Forschung nutzen

---

## Plugins

- **Plugin-Paket**: das optionale `qv_extras.py` wird separat unter [Quintet-Verify-plugin](https://github.com/Qianjinqie/Quintet-Verify-plugin) ausgeliefert – legen Sie es neben `quintet_verify.py` ab, und es wird beim Import automatisch eingebunden (keine Konfiguration nötig).  
- **Ohne das Plugin-Paket**: der Kern-Workflow läuft unverändert; die 7 Plugin-Flags bleiben einfach aus.

---

## Schnellstart

### Installation

```bash
git clone https://github.com/Qianjinqie/quintet-Verify.git
cd quintet-Verify
pip install -r requirements.txt
```

### Minimales Beispiel (echtes LLM)

```python
from quintet_verify import (
    DEFAULT_FLAGS,
    LLMConfig,
    PublicState,
    QuintetState,
    build_graph,
    build_llm_agents,
)

# 1. Modell konfigurieren (OpenAI-kompatibler Endpunkt; gemeinsam oder pro Rolle)
cfg = LLMConfig(
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
)
agents = build_llm_agents(cfg)  # liefert die fünf Rollen (A, B, C, D, E)

# 2. Graph aufbauen und ausführen (interrupt_before_e=False bedeutet: keine Pause vor E für Nutzer-Feedback)
graph = build_graph(*agents, interrupt_before_e=False)
final = graph.invoke(
    QuintetState(
        public=PublicState(
            task="Evaluate the feasibility of a technical solution and provide a conclusion.",
            user_satisfied=True,   # zufrieden → E stimmt der Entscheidung zu, Workflow endet
            intensity="high",      # Intensitätsstufen: low / high / max
            flags=dict(DEFAULT_FLAGS),  # Plugin-Schalter, alle standardmäßig aus
        )
    )
)

# 3. Ergebnisse lesen
pub = final["public"]
print(pub.verdict.approved, pub.verdict.confidence_score)  # Entscheidung und Konfidenz von D
for section in pub.draft.sections:
    print(section.anchor_id, section.content)
```

### Offline-Demo ohne LLM

```python
from quintet_verify import PublicState, QuintetState, build_graph, build_stub_agents

final = build_graph(*build_stub_agents(), interrupt_before_e=False).invoke(
    QuintetState(public=PublicState(task="Demo task", user_satisfied=True))
)
print(final["public"].verdict)
```

### Web-Konsole

```bash
python server.py  # Standardport 8000, im Browser öffnen
```

Die Konsole unterstützt: Standard-/Fast-/Custom-Pipelines, Intensitätsstufen, Plugin-Schalter, API-Konfiguration pro Rolle, Bearbeitung von Workflow-Definitionen mit Echtzeit-Validierung, Overrides für Rollen-Prompts sowie Laufzeit-Visualisierung (Event-Stream / Entwurf / Dense Track).

---

## Theoretische Ursprünge und Unabhängigkeitserklärung

### Inspirationen

Quintet-Verify schöpft Inspiration aus früheren Arbeiten in den folgenden Punkten:

· Die kognitiven Markersymbole (✓/?/✗) und die Benennung „private kognitive Spuren“ (Dense Track): abgeleitet aus der J-Space Cognition Suite V3.6 (doi:10.5281/zenodo.21971181). Jene Arbeit schlug als erste systematisch vor, kompakte Symbole innerhalb eines einzelnen Modells zur Verwaltung kognitiver Zustände zu verwenden, und belegte die positive Wirkung „introspektiver Marker“ auf die Qualität des Schlussfolgerns.
· Die Organisationsform der Multi-Rollen-Checks and Balances im Tribunal-Stil: inspiriert vom Multi-Modell-Debattensystem Quintet Verification, gezeigt auf b23.tv/z3C0vVM, das die Machbarkeit demonstrierte, „Generations-Halluzinationen durch Rollentrennung und Einspruchsverfahren in Konsensspiele zu verwandeln“.

Wir drücken unseren Respekt und unsere Dankbarkeit für die originären Beiträge der oben genannten Arbeiten aus.

### Grundlegende architektonische Unterschiede

Obwohl Quintet-Verify die Symbolbenennung von J-Space und die Organisationsform von Quintet Verification übernimmt, unterscheidet sich seine zugrunde liegende kognitive Architektur irreduzibel von beiden. Diese Unterschiede sind keine inkrementellen Verbesserungen, sondern eine Neudefinition von drei fundamentalen Annahmen: kognitives Subjekt, Informationstopologie und Art der Kontrolle.

| Aspekt | J-Space (Global-Workspace-Theorie) | Quintet Verification (Multi-Modell-Debatte) | Quintet-Verify (Anchored Cognitive Theory) |
|---|---|---|---|
| Kognitives Subjekt | Innerhalb eines einzelnen Modells | Mehrere unabhängige Modelle (zustandslos) | Mehrere unabhängige Agenten (jeweils mit privatem Zustand) |
| Träger des kognitiven Zustands | Globales Kontextfenster | Unabhängige Ausgaben jedes Modells | Öffentlicher gemeinsamer Zustand + private Spuren (physisch isoliert) |
| Informationsfluss | Globaler Broadcast + konkurrierender Zugriff | Vollständige Übertragung im Debattenstil | Verankerte Referenzen (über anchor_id + Prüfsumme für geteilte Fakten) |
| Modifikationsmechanismus | Vollständige Neuschreibung (verlässt sich auf introspektive Korrektur) | Vollständige Neuschreibung (verlässt sich auf Debattenkorrektur) | Inkrementelle Modifikation (präzise Koordinaten + Ripple-Deklarationen) |
| Art der Kontrolle | Prompt-geführte weiche Beschränkungen | Prompt-geführte weiche Beschränkungen | Validierung zur Kompilierzeit + Routing-Guards (harte Beschränkungen) |
| Wiederherstellungsmechanismus | Textuelles Ledger-Rollback | Keine standardisierte Wiederherstellung | Atomare Zustands-Checkpoints (exaktes Rollback zu jeder Runde) |
| Checks and Balances | Selbstüberwachung (gleicher Standpunkt) | Debatte über Modelle hinweg (unabhängige Standpunkte) | Berechtigungsverschränkung + Tribunalverfahren (Standpunkttrennung + Verfahrensstrenge) |
| Audierbarkeit | Chain-of-Thought-Text (nicht-strukturelles Rollback) | Debattentext (nicht-strukturelles Rollback) | Strukturiertes Dense-Track-Audit (nachverfolgbare Verifikationsbelege für ✓ und Falsifikationsbelege für ✗) |

### Zusammenfassung der Kernunterschiede

· J-Space befasst sich damit, „wie eine einzelne Person besser denkt“ (tiefe Introspektion eines einzelnen Bewusstseins);
· Quintet Verification befasst sich damit, „wie eine Gruppe durch Debatte einen Konsens erreicht“ (externe Kontrolle über mehrere Standpunkte);
· Quintet-Verify befasst sich damit, „wie eine Gruppe von Agenten mit unabhängigen kognitiven Zuständen durch starre Regeln zu überprüfbaren Schlussfolgerungen konvergiert“ (verfassungsmäßige Governance verteilter Kognition).

Der einzigartige Beitrag von Quintet-Verify ist die erstmalige Kombination von „privaten kognitiven Spuren (Dense Track)“ mit einer „starren Berechtigungsmatrix“, die es Multi-Agenten-Systemen ermöglicht, Zuverlässigkeit auf Systemebene zu erreichen – audierbar, nachverfolgbar und erzwingbar beendbar – während die individuelle Tiefe des Schlussfolgerns erhalten bleibt.

### Unabhängigkeitserklärung

Auf Basis der oben genannten grundlegenden architektonischen Unterschiede erklärt Quintet-Verify hiermit:

· Kein abgeleitetes Werk: Quintet-Verify ist kein abgeleitetes Werk und kein Fork der J-Space Cognition Suite V3.6. Die beiden verwenden völlig unterschiedliche Code-Implementierungen, kognitive Topologien und Kontrollphilosophien.
· Unabhängiges geistiges Eigentum: Der gesamte Code von Quintet-Verify wurde unabhängig von unserem Team geschrieben; kein Quellcode, keine Prompt-Vorlagen und keine Protokolltexte von J-Space oder Quintet Verification wurden kopiert oder übersetzt.
· Zitation, nicht Abhängigkeit: Dieses System führt J-Space und Quintet Verification als „theoretische Referenzen“ und nicht als „technische Abhängigkeiten“. Das Entfernen dieser Referenzen beeinträchtigt den vollen Betrieb des Kern-Workflows von Quintet-Verify nicht.
· Wissenschaftliche Integrität: Diese Erklärung zielt darauf ab, die geistigen Ursprünge korrekt nachzuzeichnen und zugleich die Grenzen unserer originären Beiträge klar zu definieren. Wir ermutigen Leser, vergleichende Studien zwischen den oben genannten Arbeiten und diesem System durchzuführen, um die unterschiedlichen Entwicklungspfade verteilter kognitiver Architekturen zu verstehen.

Empfehlung für Leser:

· Wenn Sie daran interessiert sind, „wie ein einzelnes Modell durch introspektive Marker sein Schlussfolgern verbessert“, konsultieren Sie bitte die J-Space Cognition Suite V3.6.
· Wenn Sie daran interessiert sind, „wie mehrere Modelle durch Debatte einen Konsens erreichen“, konsultieren Sie bitte das Quintet-Verification-System.
· Wenn Sie daran interessiert sind, „wie man Multi-Agenten-Kollaboration durch starre Regeln audierbare und geprüfte Schlussfolgerungen hervorbringen lässt“, fahren Sie bitte mit Quintet-Verify fort.
