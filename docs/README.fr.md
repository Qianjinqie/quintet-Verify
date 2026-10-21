[English](README.en.md) | [简体中文](../README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | **Français** | [Deutsch](README.de.md) | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)

# Quintet-Verify (QV)

**Inférence multi‑agents à contraintes strictes** · 9.17/10 en évaluation à l'aveugle · ~$0.36/exécution · Python 3.11+

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
[![Stars](https://img.shields.io/github/stars/Qianjinqie/quintet-Verify?style=social)](https://github.com/Qianjinqie/quintet-Verify/stargazers)
![Cost](https://img.shields.io/badge/cost-~$0.36%2Frun-yellowgreen)
![BlindEval](https://img.shields.io/badge/Fable%205%20Blind%20Eval-9.17%2F10-brightgreen)

Un framework de contrôle cognitif multi‑agents fondé sur une architecture cognitive ancrée – utilisant une répartition du travail à cinq rôles (A/B/C/D/E) de type tribunal et des traces cognitives privées (✓/?/✗) pour imposer des freins et contrepoids rigides aux agents au moment de l'inférence : pas de fine‑tuning, pas de modification des poids, uniquement des contraintes sur « qui parle quand, avec quelle autorité et sur la base de quelles preuves ».

- Agnostique au modèle · Python 3.11+ · LangGraph · Pydantic v2 · FastAPI

🌐 Démo en ligne (déployée via workbuddy) : https://a524b836ec13d5d9f.app.workbuddy.link

---

## Table des matières

- [Les résultats d'abord](#les-résultats-dabord)
  - [Conclusion de l'évaluation à l'aveugle par Fable 5](#fable-5-blind-evaluation-conclusion)
- [Contraintes clés](#contraintes-clés)
- [Pourquoi ne pas se contenter d'un seul grand modèle ?](#pourquoi-ne-pas-se-contenter-dun-seul-grand-modèle-)
- [Fonctionnalités principales](#fonctionnalités-principales)
- [En quoi diffère-t-il des autres frameworks multi-agents ?](#en-quoi-diffère-t-il-des-autres-frameworks-multi-agents-)
- [Explication de la qualité](#explication-de-la-qualité)
- [Données d'exécution réelle](#données-dexécution-réelle)
- [À qui s'adresse-t-il ?](#à-qui-sadresse-t-il-)
- [Plugins](#plugins)
- [Démarrage rapide](#démarrage-rapide)
  - [Installation](#installation)
  - [Exemple minimal (LLM réel)](#exemple-minimal-llm-réel)
  - [Démo hors ligne sans LLM](#démo-hors-ligne-sans-llm)
  - [Console web](#console-web)
- [Origines théoriques et déclaration d'indépendance](#origines-théoriques-et-déclaration-dindépendance)
  - [Inspirations](#inspirations)
  - [Différences architecturales fondamentales](#différences-architecturales-fondamentales)
  - [Résumé des différences essentielles](#résumé-des-différences-essentielles)
  - [Déclaration d'indépendance](#déclaration-dindépendance)

---

## Les résultats d'abord

📊 **Conclusion de l'évaluation à l'aveugle par Fable 5**

**Configuration de l'évaluation** – Quintet-Verify tourne sur DeepSeek-V4-Flash-0731 (une version légère dotée de 13 milliards de paramètres actifs ; les tâches de documents juridiques utilisent DeepSeek-V4-Pro-0813), tandis que le concurrent est le modèle d'évaluation lui-même (Fable 5), exécuté directement sans aucun framework. Le modèle évaluateur ne voit que deux étiquettes anonymes, A et B — autrement dit, sous l'évaluation d'un modèle fort, le modèle léger piloté par Quintet-Verify surpasse F5.

L'évaluation à l'aveugle par Fable 5 a attribué à notre projet une note globale de 9.17/10, tandis que le concurrent (Fable 5) a obtenu 8.0/10 – un écart de +1.17.

Les trois tâches ont toutes été remportées :

- Revue de code : 9.2 vs 8.5 (le concurrent F5 a manqué des problèmes subtils comme la perte de messages rpop, l'écrasement du champ d'état et la fuite prévisible de task_id)  
- Évaluation de solution technique : 9.5 vs 8.0 (F5 a calculé pairwise comme pointwise dans la couche de classement – une erreur fatale)  
- Revue de documents juridiques : 8.8 vs 7.5 (F5 n'a pas fourni de plan de vérification spécifique)

Remarque : la tâche de documents juridiques a utilisé DeepSeek‑V4‑Pro‑0813 ; la revue de code et l'évaluation technique ont utilisé DeepSeek‑V4‑Flash‑0731. Le framework a fonctionné en mode standard – intensité haute.

L'évaluateur (F5) a conclu qualitativement :

> Le système évalué surpasse le sujet sur cinq dimensions : profondeur technique, exactitude des calculs, identification des risques, appui sur les données et rigueur professionnelle. Convient aux scénarios à enjeux élevés tels que les décisions d'investissement, la due diligence technique, les revues de déploiement avant mise en production et les audits de conformité.

Remarque : en pratique, QV effectue de nombreux appels API, ce qui entraîne des coûts plus élevés – il échange de l'argent et du temps contre de la qualité.

---

## Contraintes clés

- A n'écrit pas le texte principal, B ne modifie pas le framework, C ne fournit pas de révisions du texte intégral, D ne rédige pas à la place des autres, E ne renverse pas directement les décisions.  
- Les marqueurs cognitifs ✓/?/✗ existent sous forme de traces privées et n'entrent jamais dans les décisions de routage.  
- Les renvois doivent inclure des coordonnées d'ancrage ; les retouches sont limitées aux régions désignées (les déclarations de ripple bornent la portée des modifications).

---

## Pourquoi ne pas se contenter d'un seul grand modèle ?

Chez un modèle unique, « la génération vaut verdict » est la cause racine des hallucinations qui atteignent directement le point final : il génère, relit et confirme tout seul – le superviseur et le supervisé partagent le même point de vue. Quintet‑Verify brise cette boucle fermée en cinq rôles aux permissions mutuellement verrouillées –

Les freins et contrepoids passent de l'« autodiscipline » à l'« hétérodiscipline » : le superviseur et le supervisé ne partagent pas le même point de vue ; chaque renvoi est confiné dans les coordonnées d'ancrage, si bien que les retouches ne font jamais dériver l'ensemble du texte.

---

## Fonctionnalités principales

- **Architecture cognitive à double canal** : le canal public (visible du routage) et la Dense Track privée (✓/?/✗) sont physiquement isolés, avec trois couches de protection (structurelle / typage / assertions à l'exécution) garantissant que la cognition privée n'entre jamais dans le routage.  
- **Freins et contrepoids rigides à cinq rôles** : A (Architecte) · B (Exécutant) · C (Vérificateur) · D (Juge) · E (Juré) – chacun avec des permissions verrouillées et des zones interdites.  
- **Trois règles de fer (appliquées dans le code)** : ① isolement de la confiance ② modification incrémentale (coordonnées d'ancrage + déclarations de ripple) ③ disjoncteur et injonction de silence (max_iterations : 1–10, 5 par défaut ; D peut prononcer une injonction de silence).  
- **Niveaux d'intensité** : low / high (par défaut) / max ; en mode strict, chaque défaut doit être accompagné d'une chaîne d'impact, et D ne peut pas valider si la confiance est insuffisante.  
- **Mode rapide** : uniquement le triangle B/C/D, pas de plan, pas de branche d'appel – adapté aux tâches simples et aux scénarios à faible coût.  
- **Gouvernance du surapprentissage (overfitting)** : C n'a aucun quota obligatoire (zéro défaut est permis, mais doit s'accompagner d'une déclaration de couverture) ; D peut juger C excessivement pointilleux et le renvoyer pour relecture (2 fois maximum).  
- **Mécanisme de plugins** : registre PLUGINS + flags (tous désactivés par défaut) ; le qv_extras.py optionnel peut être monté via des pre/post_act_hooks standardisés – voir [Plugins](#plugins).  
- **Workflow personnalisable (v1.4)** : édition front‑end des définitions de workflow, création de nouveaux rôles héritant des permissions de A et E, validation statique et configuration LLM indépendante par rôle.

---

## En quoi diffère-t-il des autres frameworks multi-agents ?

- **AutoGen / CrewAI / MetaGPT** : tous prennent en charge la collaboration multi‑agents et sont agnostiques au modèle, mais il leur manque trois capacités : les freins et contrepoids rigides au moment de l'inférence, l'audit des traces cognitives privées et la modification incrémentale (économie de tokens).  
- **Quintet‑Verify** : tout en prenant en charge la collaboration multi‑agents et l'agnosticisme au modèle, il fournit en supplément :  
  - Freins et contrepoids rigides au moment de l'inférence (verrouillage des permissions de type tribunal)  
  - Traces cognitives privées auditables (Dense Track)  
  - Modification incrémentale (coordonnées d'ancrage + déclarations de ripple, économisant en moyenne ~80 % de contexte)

La plupart des frameworks font « laisser l'IA collaborer » ; Quintet‑Verify fait « laisser l'IA se freiner et se contrôler mutuellement » – le premier recherche l'efficacité de la sortie, le second la fiabilité des conclusions.

---

## Explication de la qualité

Par rapport à l'utilisation directe d'un modèle unique, les gains de Quintet‑Verify proviennent des freins et contrepoids structurels, et non d'un plus grand nombre de paramètres :

- **Par rapport à un modèle pur** : les hallucinations n'atteignent plus directement le point final – elles doivent d'abord passer la relecture de C puis la décision de D, et chaque renvoi est confiné dans les coordonnées d'ancrage. La conclusion « zéro défaut » de C et le jugement de surapprentissage de D empêchent en même temps que le frottement interne ne dégénère en « ergoter pour ergoter ».  
- **Par rapport à du J‑Space pur** : la gestion cognitive dense de J‑Space s'opère à l'intérieur d'un modèle unique ; le plafond de l'autosupervision est « se détecter soi‑même ». Quintet‑Verify conserve ✓/?/✗ comme traces cognitives privées tout en déléguant l'autorité de décision à des rôles indépendants – faisant passer les freins et contrepoids de l'autodiscipline à l'hétérodiscipline.

Ce qui précède est une comparaison qualitative au niveau des mécanismes ; en pratique, l'efficacité est généralement bien supérieure à celle de son modèle de base – voir Les résultats d'abord pour plus de détails.

---

## Données d'exécution réelle

Exécution réelle en mode complet (août 2026, DeepSeek‑V4‑Pro‑0813, tâche de revue de documents juridiques) :

- Consommation de tokens : ~130,000  
- Appels API : 19  
- Coût : ~¥2.3 (~$0.36)  
- Décision finale : D a approuvé avec une confiance de 90/100

La sortie a conclu « faisable sous conditions avec une architecture hybride » : les affirmations empiriques dépourvues de données à l'appui dans l'ébauche initiale ont été soit entièrement supprimées, soit rétrogradées en hypothèses en attente de vérification après deux renvois ; les critères d'acceptation ont été réécrits en termes auditables.

---

## À qui s'adresse-t-il ?

- Les développeurs qui construisent des applications RAG mais en proie aux hallucinations  
- Les scénarios juridiques, financiers et de santé qui exigent des décisions d'IA auditables  
- Les chercheurs qui veulent expérimenter les freins et contrepoids multi‑agents sans écrire d'orchestrations complexes  
- Les équipes satisfaites de l'efficacité de collaboration d'AutoGen/CrewAI mais exigeant une meilleure fiabilité des conclusions  
- Les programmeurs qui utilisent l'IA pour écrire du code de haute qualité  
- Les chercheurs qui utilisent l'IA comme assistance à la recherche mathématique

---

## Plugins

- **Pack de plugins** : le qv_extras.py optionnel est distribué séparément sur [Quintet-Verify-plugin](https://github.com/Qianjinqie/Quintet-Verify-plugin) – déposez-le à côté de quintet_verify.py et il est monté automatiquement à l'import (aucune configuration nécessaire).  
- **Sans le pack de plugins** : le workflow central fonctionne à l'identique ; les 7 flags de plugins restent simplement désactivés.

---

## Démarrage rapide

### Installation

```bash
git clone https://github.com/Qianjinqie/quintet-Verify.git
cd quintet-Verify
pip install -r requirements.txt
```

### Exemple minimal (LLM réel)

```python
from quintet_verify import (
    DEFAULT_FLAGS,
    LLMConfig,
    PublicState,
    QuintetState,
    build_graph,
    build_llm_agents,
)

# 1. Configurer le modèle (endpoint compatible OpenAI ; partagé ou par rôle)
cfg = LLMConfig(
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
)
agents = build_llm_agents(cfg)  # renvoie les cinq rôles (A, B, C, D, E)

# 2. Construire le graphe et exécuter (interrupt_before_e=False signifie pas de pause avant E pour recueillir un retour utilisateur)
graph = build_graph(*agents, interrupt_before_e=False)
final = graph.invoke(
    QuintetState(
        public=PublicState(
            task="Evaluate the feasibility of a technical solution and provide a conclusion.",
            user_satisfied=True,   # satisfait → E approuve la décision, le workflow se termine
            intensity="high",      # niveaux d'intensité : low / high / max
            flags=dict(DEFAULT_FLAGS),  # commutateurs de plugins, tous désactivés par défaut
        )
    )
)

# 3. Lire les résultats
pub = final["public"]
print(pub.verdict.approved, pub.verdict.confidence_score)  # décision de D et niveau de confiance
for section in pub.draft.sections:
    print(section.anchor_id, section.content)
```

### Démo hors ligne sans LLM

```python
from quintet_verify import PublicState, QuintetState, build_graph, build_stub_agents

final = build_graph(*build_stub_agents(), interrupt_before_e=False).invoke(
    QuintetState(public=PublicState(task="Demo task", user_satisfied=True))
)
print(final["public"].verdict)
```

### Console web

```bash
python server.py  # port par défaut : 8000, à ouvrir dans le navigateur
```

La console prend en charge : les pipelines Standard / Fast / Custom, les niveaux d'intensité, les bascules de plugins, la configuration API par rôle, l'édition des définitions de workflow avec validation en temps réel, la surcharge des prompts de rôles et la visualisation des exécutions (flux d'événements / ébauche / Dense Track).

---

## Origines théoriques et déclaration d'indépendance

### Inspirations

Quintet‑Verify s'inspire des travaux antérieurs sur les aspects suivants :

· Les symboles de marqueurs cognitifs (✓/?/✗) et la dénomination de « traces cognitives privées » (Dense Track) : issus de la J‑Space Cognition Suite V3.6 (doi:10.5281/zenodo.21971181). Ce travail a le premier proposé, de manière systématique, d'utiliser des symboles compacts à l'intérieur d'un modèle unique pour gérer les états cognitifs, et a validé l'effet positif des « marqueurs d'introspection » sur la qualité du raisonnement.
· La forme d'organisation des freins et contrepoids multi‑rôles de type tribunal : inspirée du système de débat multi‑modèles Quintet Verification présenté sur b23.tv/z3C0vVM, qui a démontré la faisabilité de « transformer les hallucinations de génération en jeux de consensus par la séparation des rôles et des procédures d'objection ».

Nous exprimons notre respect et notre gratitude pour les contributions originales des œuvres susmentionnées.

### Différences architecturales fondamentales

Bien que Quintet‑Verify emprunte la dénomination des symboles à J‑Space et la forme d'organisation à Quintet Verification, son architecture cognitive sous‑jacente diffère de manière irréductible de l'une comme de l'autre. Ces différences ne sont pas des améliorations incrémentales, mais une redéfinition de trois hypothèses fondamentales : le sujet cognitif, la topologie de l'information et la nature du contrôle.

| Aspect | J‑Space (Global Workspace Theory) | Quintet Verification (débat multi‑modèles) | Quintet‑Verify (Anchored Cognitive Theory) |
|---|---|---|---|
| Sujet cognitif | À l'intérieur d'un modèle unique | Plusieurs modèles indépendants (sans état) | Plusieurs agents indépendants (chacun avec un état privé) |
| Porteur de l'état cognitif | Fenêtre de contexte globale | Sorties indépendantes de chaque modèle | État partagé public + traces privées (physiquement isolées) |
| Flux d'information | Diffusion globale + accès concurrentiel | Transmission intégrale de type débat | Références ancrées (via anchor_id + checksum pour les faits partagés) |
| Mécanisme de modification | Réécriture complète (repose sur la correction introspective) | Réécriture complète (repose sur la correction par débat) | Modification incrémentale (coordonnées précises + déclarations de ripple) |
| Nature du contrôle | Contraintes souples guidées par prompts | Contraintes souples guidées par prompts | Validation à la compilation + gardes de routage (contraintes strictes) |
| Mécanisme de récupération | Rollback d'un registre textuel | Pas de récupération standardisée | Checkpoints d'état atomiques (rollback exact vers n'importe quel tour) |
| Freins et contrepoids | Autosupervision (même point de vue) | Débat entre modèles (points de vue indépendants) | Verrouillage des permissions + procédure de tribunal (séparation des points de vue + rigidité procédurale) |
| Auditabilité | Texte de chaîne de pensée (rollback non structurel) | Texte de débat (rollback non structurel) | Audit structuré de la Dense Track (preuves de vérification traçables pour ✓ et preuves de falsification pour ✗) |

### Résumé des différences essentielles

· J‑Space traite de « comment une personne seule peut mieux penser » (introspection profonde d'une conscience unique) ;
· Quintet Verification traite de « comment un groupe parvient à un consensus par le débat » (contrôle externe via plusieurs points de vue) ;
· Quintet‑Verify traite de « comment un groupe d'agents dotés d'états cognitifs indépendants converge vers des conclusions vérifiables au moyen de règles rigides » (gouvernance constitutionnelle de la cognition distribuée).

La contribution unique de Quintet‑Verify est la première combinaison des « traces cognitives privées (Dense Track) » avec une « matrice de permissions rigide », permettant aux systèmes multi‑agents d'atteindre une fiabilité au niveau système — auditable, traçable et terminable de force — tout en préservant la profondeur de raisonnement individuelle.

### Déclaration d'indépendance

Sur la base des différences architecturales fondamentales ci‑dessus, Quintet‑Verify déclare par la présente :

· Pas une œuvre dérivée : Quintet‑Verify n'est ni une œuvre dérivée ni un fork de la J‑Space Cognition Suite V3.6. Les deux adoptent des implémentations de code, des topologies cognitives et des philosophies de contrôle complètement différentes.
· Propriété intellectuelle indépendante : tout le code de Quintet‑Verify a été écrit de manière indépendante par notre équipe ; aucun code source, aucun template de prompt ni aucun texte de protocole provenant de J‑Space ou de Quintet Verification n'a été copié ni traduit.
· Citation, pas dépendance : ce système cite J‑Space et Quintet Verification comme « références théoriques » plutôt que comme « dépendances techniques ». La suppression de ces références n'affecte pas le fonctionnement complet du workflow central de Quintet‑Verify.
· Intégrité académique : cette déclaration vise à retracer fidèlement les origines intellectuelles tout en délimitant clairement les frontières de nos contributions originales. Nous encourageons les lecteurs à mener des études comparatives entre les œuvres susmentionnées et ce système, afin de comprendre les différentes trajectoires évolutives des architectures cognitives distribuées.

Recommandation à l'attention des lecteurs :

· Si les « améliorations du raisonnement d'un modèle unique par des marqueurs d'introspection » vous intéressent, reportez‑vous à la J‑Space Cognition Suite V3.6.
· Si la « convergence de plusieurs modèles vers un consensus par le débat » vous intéresse, reportez‑vous au système Quintet Verification.
· Si la façon de « faire produire à la collaboration multi‑agents des conclusions auditables et contrôlées au moyen de règles rigides » vous intéresse, continuez d'utiliser Quintet‑Verify.
