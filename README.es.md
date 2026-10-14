[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | **Español** | [Français](README.fr.md) | [Deutsch](README.de.md) | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)</p>

# Quintet-Verify (QV)

**Inferencia multiagente con restricciones estrictas** · 9.17/10 en evaluación a ciegas · ~$0.36/ejecución · Python 3.11+

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
[![Stars](https://img.shields.io/github/stars/Qianjinqie/quintet-Verify?style=social)](https://github.com/Qianjinqie/quintet-Verify/stargazers)
![Cost](https://img.shields.io/badge/cost-~$0.36%2Frun-yellowgreen)
![BlindEval](https://img.shields.io/badge/Fable%205%20Blind%20Eval-9.17%2F10-brightgreen)

Un framework de control cognitivo multiagente basado en una arquitectura cognitiva anclada – emplea una división del trabajo estilo tribunal con cinco roles (A/B/C/D/E) y trazas cognitivas privadas (✓/?/✗) para imponer frenos y contrapesos rígidos a los agentes en el momento de la inferencia: sin fine‑tuning, sin cambios de pesos, solo restricciones sobre “quién habla cuándo, con qué autoridad y con base en qué evidencia.”

- Independiente del modelo · Python 3.11+ · LangGraph · Pydantic v2 · FastAPI

🌐 Demostración en línea (desplegada vía workbuddy): https://a524b836ec13d5d9f.app.workbuddy.link

---

## Tabla de contenidos

- [Resultados primero](#resultados-primero)
  - [Conclusión de la evaluación a ciegas de Fable 5](#conclusión-de-la-evaluación-a-ciegas-de-fable-5)
- [Restricciones clave](#restricciones-clave)
- [¿Por qué no usar simplemente un modelo grande?](#por-qué-no-usar-simplemente-un-modelo-grande)
- [Características principales](#características-principales)
- [¿En qué se diferencia de otros frameworks multiagente?](#en-qué-se-diferencia-de-otros-frameworks-multiagente)
- [Explicación de la calidad](#explicación-de-la-calidad)
- [Datos de ejecución real](#datos-de-ejecución-real)
- [¿Quién debería usarlo?](#quién-debería-usarlo)
- [Plugins](#plugins)
- [Inicio rápido](#inicio-rápido)
  - [Instalación](#instalación)
  - [Ejemplo mínimo (LLM real)](#ejemplo-mínimo-llm-real)
  - [Demostración offline sin LLM](#demostración-offline-sin-llm)
  - [Consola web](#consola-web)
- [Orígenes teóricos y declaración de independencia](#orígenes-teóricos-y-declaración-de-independencia)
  - [Inspiraciones](#inspiraciones)
  - [Diferencias arquitectónicas fundamentales](#diferencias-arquitectónicas-fundamentales)
  - [Resumen de las diferencias principales](#resumen-de-las-diferencias-principales)
  - [Declaración de independencia](#declaración-de-independencia)

---

## Resultados primero

📊 **Conclusión de la evaluación a ciegas de Fable 5**

**Configuración de la evaluación** – Quintet-Verify se ejecuta sobre DeepSeek-V4-Flash-0731 (una versión ligera con 13 mil millones de parámetros activos; las tareas de documentos legales usan DeepSeek-V4-Pro-0813), mientras que el contrincante es el propio modelo de evaluación (Fable 5), ejecutándose directamente sin ningún framework. El modelo de evaluación solo ve dos etiquetas anónimas, A y B — en otras palabras, bajo la puntuación de un modelo fuerte, el modelo ligero impulsado por Quintet-Verify supera a F5.

La evaluación a ciegas de Fable 5 otorgó a nuestro proyecto una puntuación global de 9.17/10, mientras que el contrincante (Fable 5) recibió 8.0/10 – un margen de +1.17.

Se ganaron las tres tareas:

- Revisión de código: 9.2 vs 8.5 (el contrincante F5 pasó por alto problemas sutiles como la pérdida de mensajes de rpop, la sobrescritura del campo de estado y la fuga predecible de task_id)  
- Evaluación de soluciones técnicas: 9.5 vs 8.0 (F5 calculó erróneamente pairwise como pointwise en la capa de ranking – un error fatal)  
- Revisión de documentos legales: 8.8 vs 7.5 (F5 no proporcionó un plan de verificación específico)

Nota: la tarea de documentos legales usó DeepSeek‑V4‑Pro‑0813; la revisión de código y la evaluación técnica usaron DeepSeek‑V4‑Flash‑0731. El framework se ejecutó en modo estándar – intensidad alta.

El evaluador (F5) concluyó cualitativamente:

> El sistema evaluado supera al sujeto en cinco dimensiones: profundidad técnica, precisión computacional, identificación de riesgos, respaldo de datos y rigor profesional. Adecuado para escenarios de alto riesgo como decisiones de inversión, due diligence técnica, revisiones de despliegue en preproducción y auditorías de cumplimiento.

Nota: en la práctica, QV realiza muchas llamadas a la API, lo que genera costos más altos – cambia dinero y tiempo por calidad.

---

## Restricciones clave

- A no escribe el texto principal, B no altera el framework, C no proporciona revisiones del texto completo, D no hace ghostwriting en nombre de otros, E no revoca directamente los fallos.  
- Los marcadores cognitivos ✓/?/✗ existen como trazas privadas y nunca entran en las decisiones de enrutamiento.  
- Las devoluciones deben incluir coordenadas de anclaje; la reelaboración se limita a las regiones designadas (las declaraciones de ripple acotan el alcance de los cambios).

---

## ¿Por qué no usar simplemente un modelo grande?

El “generar equivale a sentenciar” de un modelo único es la causa raíz de que las alucinaciones lleguen directamente al resultado final: él mismo genera, revisa y confirma – el supervisor y el supervisado comparten el mismo punto de vista. Quintet‑Verify rompe este circuito cerrado en cinco roles con permisos bloqueados entre sí –

Los frenos y contrapesos evolucionan de la “autodisciplina” a la “heterodisciplina”: el supervisor y el supervisado no comparten un punto de vista; cada devolución queda constreñida dentro de coordenadas de anclaje, de modo que la reelaboración nunca provoca que el texto completo derive.

---

## Características principales

- **Arquitectura cognitiva de doble canal**: el canal público (visible para el enrutamiento) y el Dense Track privado (✓/?/✗) están físicamente aislados, con tres capas de protección (estructural / de tipos / aserciones en tiempo de ejecución) que garantizan que la cognición privada nunca entra en el enrutamiento.  
- **Frenos y contrapesos rígidos entre cinco roles**: A (Arquitecto) · B (Ejecutor) · C (Verificador) · D (Juez) · E (Jurado) – cada uno con permisos bloqueados y zonas prohibidas.  
- **Tres reglas de hierro (aplicadas en código)**: ① aislamiento de confianza ② modificación incremental (coordenadas de anclaje + declaraciones de ripple) ③ orden de silencio tipo circuit‑breaker (max_iterations: 1–10, por defecto 5; D puede emitir una orden de silencio).  
- **Niveles de intensidad**: low / high (por defecto) / max; en modo estricto, los defectos deben ir acompañados de una cadena de impacto, y D no puede aprobar si la confianza es insuficiente.  
- **Modo Fast**: solo el triángulo B/C/D, sin esquema, sin rama de apelación – adecuado para tareas simples y escenarios de bajo costo.  
- **Gobernanza del overfitting**: C no tiene una cuota obligatoria (se permiten cero defectos, pero debe incluir una declaración de cobertura); D puede considerar que C es excesivamente pedante y devolverlo a revisión (máx. 2 veces).  
- **Mecanismo de plugins**: registro PLUGINS + flags (todos desactivados por defecto); el qv_extras.py opcional se puede montar mediante pre/post_act_hooks estandarizados – ver [Plugins](#plugins).  
- **Workflow personalizable (v1.4)**: edición front‑end de las definiciones de workflow, creación de nuevos roles que heredan los permisos AE, validación estática y configuración LLM independiente por rol.

---

## ¿En qué se diferencia de otros frameworks multiagente?

- **AutoGen / CrewAI / MetaGPT**: todos admiten colaboración multiagente y son independientes del modelo, pero carecen de tres capacidades: frenos y contrapesos rígidos en tiempo de inferencia, auditoría de trazas cognitivas privadas y modificación incremental (ahorro de tokens).  
- **Quintet‑Verify**: además de admitir colaboración multiagente e independencia del modelo, proporciona adicionalmente:  
  - Frenos y contrapesos rígidos en tiempo de inferencia (entrelazamiento de permisos estilo tribunal)  
  - Trazas cognitivas privadas auditables (Dense Track)  
  - Modificación incremental (coordenadas de anclaje + declaraciones de ripple, ahorrando en promedio ~80% de contexto)

La mayoría de los frameworks hacen “que la IA colabore”; Quintet‑Verify hace “que la IA ejerza frenos y contrapesos entre sí” – los primeros persiguen la eficiencia de salida, el segundo persigue la confiabilidad de las conclusiones.

---

## Explicación de la calidad

En comparación con usar un modelo único directamente, las ganancias de Quintet‑Verify provienen de los frenos y contrapesos estructurales, no de un mayor número de parámetros:

- **Comparado con un modelo puro**: las alucinaciones ya no llegan directamente al resultado final – primero deben pasar por la revisión de C y el fallo de D, y cada devolución queda constreñida dentro de coordenadas de anclaje. La conclusión de cero defectos de C y el juicio de overfitting de D evitan a la vez la fricción interna de “buscar defectos por buscar”.  
- **Comparado con J‑Space puro**: la gestión cognitiva densa de J‑Space opera dentro de un modelo único; el techo de la autosupervisión es “atraparse a uno mismo”. Quintet‑Verify conserva ✓/?/✗ como trazas cognitivas privadas mientras delega la autoridad de fallo en roles independientes – eleva los frenos y contrapesos de la autodisciplina a la heterodisciplina.

Lo anterior es una comparación cualitativa a nivel de mecanismo; en la práctica, la efectividad suele ser muy superior a la de su modelo base – ver Resultados primero para más detalles.

---

## Datos de ejecución real

Ejecución real en modo completo (agosto de 2026, DeepSeek‑V4‑Pro‑0813, tarea de revisión de documentos legales):

- Consumo de tokens: ~130,000  
- Llamadas a la API: 19  
- Costo: ~¥2.3 (~$0.36)  
- Fallo final: D aprobó con una confianza de 90/100

El resultado concluyó “condicionalmente factible con arquitectura híbrida”: las afirmaciones empíricas sin respaldo de datos presentes en el borrador inicial fueron eliminadas por completo o degradadas a hipótesis pendientes de verificación tras dos devoluciones; los criterios de aceptación fueron reescritos en términos auditables.

---

## ¿Quién debería usarlo?

- Desarrolladores que construyen aplicaciones RAG pero están plagados de alucinaciones  
- Escenarios legales, financieros y de salud que requieren decisiones de IA auditables  
- Investigadores que quieren experimentar los frenos y contrapesos multiagente sin escribir orquestaciones complejas  
- Equipos satisfechos con la eficiencia de colaboración de AutoGen/CrewAI pero que exigen una mayor confiabilidad de las conclusiones  
- Programadores que usan IA para escribir código de alta calidad  
- Investigadores que usan IA como asistencia para la investigación matemática

---

## Plugins

- **Paquete de plugins**: el `qv_extras.py` opcional se distribuye por separado en [Quintet-Verify-plugin](https://github.com/Qianjinqie/Quintet-Verify-plugin) – colócalo junto a `quintet_verify.py` y se montará automáticamente al momento de importar (sin necesidad de configuración).  
- **Sin el paquete de plugins**: el workflow principal se ejecuta sin cambios; los 7 flags de plugins simplemente permanecen desactivados.

---

## Inicio rápido

### Instalación

```bash
git clone https://github.com/Qianjinqie/quintet-Verify.git
cd quintet-Verify
pip install -r requirements.txt
```

### Ejemplo mínimo (LLM real)

```python
from quintet_verify import (
    DEFAULT_FLAGS,
    LLMConfig,
    PublicState,
    QuintetState,
    build_graph,
    build_llm_agents,
)

# 1. Configurar el modelo (endpoint compatible con OpenAI; compartido o por rol)
cfg = LLMConfig(
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
)
agents = build_llm_agents(cfg)  # devuelve los cinco roles (A, B, C, D, E)

# 2. Construir el grafo y ejecutar (interrupt_before_e=False significa que no hay pausa antes de E para el feedback del usuario)
graph = build_graph(*agents, interrupt_before_e=False)
final = graph.invoke(
    QuintetState(
        public=PublicState(
            task="Evaluate the feasibility of a technical solution and provide a conclusion.",
            user_satisfied=True,   # satisfecho → E está de acuerdo con el fallo, el workflow termina
            intensity="high",      # niveles de intensidad: low / high / max
            flags=dict(DEFAULT_FLAGS),  # interruptores de plugins, todos desactivados por defecto
        )
    )
)

# 3. Leer los resultados
pub = final["public"]
print(pub.verdict.approved, pub.verdict.confidence_score)  # el fallo de D y la confianza
for section in pub.draft.sections:
    print(section.anchor_id, section.content)
```

### Demostración offline sin LLM

```python
from quintet_verify import PublicState, QuintetState, build_graph, build_stub_agents

final = build_graph(*build_stub_agents(), interrupt_before_e=False).invoke(
    QuintetState(public=PublicState(task="Demo task", user_satisfied=True))
)
print(final["public"].verdict)
```

### Consola web

```bash
python server.py  # puerto por defecto 8000, abrir en el navegador
```

La consola admite: pipelines Standard / Fast / Custom, niveles de intensidad, interruptores de plugins, configuración de API por rol, edición de definiciones de workflow y validación en tiempo real, sobrescritura de los prompts de rol y visualización de ejecuciones (flujo de eventos / borrador / Dense Track).

---

## Orígenes teóricos y declaración de independencia

### Inspiraciones

Quintet‑Verify se inspira en trabajos previos en los siguientes aspectos:

· Los símbolos de marcadores cognitivos (✓/?/✗) y la denominación de “trazas cognitivas privadas” (Dense Track): provienen del J‑Space Cognition Suite V3.6 (doi:10.5281/zenodo.21971181). Ese trabajo fue el primero en proponer sistemáticamente el uso de símbolos compactos dentro de un modelo único para gestionar estados cognitivos, y validó el efecto positivo de los “marcadores introspectivos” sobre la calidad del razonamiento.
· La forma organizativa de los frenos y contrapesos estilo tribunal multi‑rol: inspirada en el sistema de debate multi‑modelo Quintet Verification mostrado en b23.tv/z3C0vVM, que demostró la viabilidad de “transformar las alucinaciones de generación en juegos de consenso mediante la separación de roles y procedimientos de objeción”.

Expresamos nuestro respeto y gratitud por las contribuciones originales de los trabajos mencionados.

### Diferencias arquitectónicas fundamentales

Aunque Quintet‑Verify toma prestada la denominación de los símbolos de J‑Space y la forma organizativa de Quintet Verification, su arquitectura cognitiva subyacente difiere de ambos de manera irreducible. Estas diferencias no son mejoras incrementales, sino una redefinición de tres supuestos fundamentales: el sujeto cognitivo, la topología de la información y la naturaleza del control.

| Aspecto | J‑Space (Teoría del Espacio de Trabajo Global) | Quintet Verification (debate multi‑modelo) | Quintet‑Verify (Teoría Cognitiva Anclada) |
|---|---|---|---|
| Sujeto cognitivo | Dentro de un modelo único | Múltiples modelos independientes (sin estado) | Múltiples agentes independientes (cada uno con estado privado) |
| Portador del estado cognitivo | Ventana de contexto global | Salidas independientes de cada modelo | Estado compartido público + trazas privadas (físicamente aisladas) |
| Flujo de información | Difusión global + acceso competitivo | Transmisión completa estilo debate | Referencias ancladas (vía anchor_id + checksum para hechos compartidos) |
| Mecanismo de modificación | Reescritura completa (depende de la corrección introspectiva) | Reescritura completa (depende de la corrección por debate) | Modificación incremental (coordenadas precisas + declaraciones de ripple) |
| Naturaleza del control | Restricciones blandas guiadas por prompts | Restricciones blandas guiadas por prompts | Validación en tiempo de compilación + guardas de enrutamiento (restricciones duras) |
| Mecanismo de recuperación | Reversión mediante registro textual | Sin recuperación estandarizada | Checkpoints de estado atómicos (reversión exacta a cualquier ronda) |
| Frenos y contrapesos | Autosupervisión (mismo punto de vista) | Debate entre modelos (puntos de vista independientes) | Entrelazamiento de permisos + procedimiento de tribunal (separación de puntos de vista + rigidez procedimental) |
| Auditabilidad | Texto de chain‑of‑thought (reversión no estructural) | Texto de debate (reversión no estructural) | Auditoría estructurada del Dense Track (evidencia de verificación trazable para ✓ y evidencia de falsación para ✗) |

### Resumen de las diferencias principales

· J‑Space aborda “cómo una sola persona piensa mejor” (introspección profunda de una única conciencia);
· Quintet Verification aborda “cómo un grupo alcanza el consenso mediante el debate” (verificación externa a través de múltiples puntos de vista);
· Quintet‑Verify aborda “cómo un grupo de agentes con estados cognitivos independientes converge en conclusiones verificables mediante reglas rígidas” (gobernanza constitucional de la cognición distribuida).

La contribución única de Quintet‑Verify es la primera combinación de “trazas cognitivas privadas (Dense Track)” con una “matriz de permisos rígida”, que permite a los sistemas multiagente lograr una confiabilidad a nivel de sistema auditable, trazable y terminable de forma forzada, manteniendo a la vez la profundidad de razonamiento individual.

### Declaración de independencia

Con base en las diferencias arquitectónicas fundamentales anteriores, Quintet‑Verify declara por la presente:

· No es una obra derivada: Quintet‑Verify no es una obra derivada ni un fork del J‑Space Cognition Suite V3.6. Ambos adoptan implementaciones de código, topologías cognitivas y filosofías de control completamente diferentes.
· Propiedad intelectual independiente: todo el código de Quintet‑Verify fue escrito de forma independiente por nuestro equipo; no se ha copiado ni traducido ningún código fuente, plantilla de prompts ni texto de protocolo de J‑Space o Quintet Verification.
· Citación, no dependencia: este sistema enumera a J‑Space y Quintet Verification como “referencias teóricas” y no como “dependencias técnicas”. Eliminar estas referencias no afecta el funcionamiento completo del workflow principal de Quintet‑Verify.
· Integridad académica: esta declaración busca trazar con precisión los orígenes intelectuales a la vez que define con claridad los límites de nuestras contribuciones originales. Animamos a los lectores a realizar estudios comparativos entre los trabajos mencionados y este sistema para comprender las distintas rutas evolutivas de las arquitecturas cognitivas distribuidas.

Recomendación para los lectores:

· Si te interesa “cómo un modelo único mejora el razonamiento mediante marcadores introspectivos”, consulta el J‑Space Cognition Suite V3.6.
· Si te interesa “cómo múltiples modelos alcanzan el consenso mediante el debate”, consulta el sistema Quintet Verification.
· Si te interesa “cómo hacer que la colaboración multiagente produzca conclusiones auditables y contrastadas mediante reglas rígidas”, sigue usando Quintet‑Verify.
