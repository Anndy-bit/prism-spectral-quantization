# PRISM — Precision Robustness in Spectral vs. additive Modulation

## Idea en una frase

Cuando fusionás un adaptador PEFT en los pesos congelados y después cuantizás el modelo resultante para correrlo en hardware barato (CPU, celular), ¿importa **qué tipo** de adaptador usaste? Nuestra hipótesis: un adaptador espectral acotado (SVMO, de S³) sobrevive la cuantización mejor que uno aditivo sin restricción (LoRA), porque por construcción no puede crear outliers nuevos en la distribución de pesos.

Esto es un proyecto hermano de `lowrank-field-adapters` (S³/USF), no una continuación — usa los checkpoints que ya existen ahí, pero la pregunta de investigación es distinta y autocontenida.

## Por qué esto y no otra cosa

Después de revisar a fondo qué datos y resultados reales tiene el laboratorio (ver el repo `lowrank-field-adapters`), la conclusión fue: no hay un tercer paper agarrable exprimiendo más los datos de entrenamiento que ya existen — todo lo real ya está reclamado por S³ o por USF. Pero hay un ángulo que **no requiere entrenar nada nuevo**: la fase de *inferencia/despliegue* de lo que ya entrenamos, mirada con una pregunta que nadie en la literatura reciente de cuantización parece haber hecho.

Esto también resuelve un problema práctico inmediato: la GPU actual disponible (GeForce GT 710, Kepler, compute capability 3.5) **no es utilizable con las herramientas modernas** — CUDA 12.x y PyTorch 2.x dejaron de dar soporte a Kepler. Este proyecto no necesita esa GPU en absoluto: todo el trabajo central es CPU, y el despliegue final es en un Android.

## La pregunta de investigación

**¿La geometría de la actualización de pesos de un adaptador PEFT determina qué tan bien sobrevive la cuantización post-entrenamiento?**

Dos familias de adaptador, mismo modelo base (Qwen2.5-7B-Instruct), ya entrenados en el laboratorio:

| | LoRA | SVMO (de S³) |
|---|---|---|
| Forma de la actualización | $W' = W + BA$ (aditiva, sin cota) | $W' = W + U_k(m_\theta(\Sigma_k) - \Sigma_k)V_k^T$ (multiplicativa en el dominio espectral) |
| Rango de la modulación | Sin restricción — $BA$ puede tener cualquier magnitud/dirección | Acotada: $m_\theta(\sigma) \in [\sigma(1-\alpha), \sigma(1+\alpha)]$, $\alpha=0.3$, saturada por $\tanh$ (Teorema III.4 de S³, ya demostrado) |
| ¿Puede crear una dirección de peso completamente nueva? | Sí | No — solo reescala direcciones singulares que ya existían en $W$ |

**Hipótesis (falsable):** al fusionar cada adaptador en $W$ y cuantizar a 4 bits (o menos), el SVMO-merged debería mostrar menor degradación de perplejidad y menor kurtosis/outlier-ratio por canal que el LoRA-merged, porque el Teorema III.4 ya prueba que la modulación es numéricamente bien condicionada — la cuantización penaliza justo lo que SVMO estructuralmente no puede producir.

**Esto no es una garantía — es una hipótesis con motivo teórico real, que se mide, no se asume.** Si el resultado sale al revés o no hay diferencia significativa, eso también es publicable (negative result honesto), no invalida el proyecto.

## Related work real (verificado hoy, no de memoria)

- **AWQ** (Activation-aware Weight Quantization) y **SpinQuant** (Meta, rotaciones aprendidas) existen específicamente porque las actualizaciones de pesos introducen outliers que dañan la cuantización naive — es decir, la industria ya reconoce que "cómo" se llega a los pesos importa para cuantizar, pero nadie parece haber comparado explícitamente adaptadores espectrales acotados contra aditivos bajo esta lupa.
- **GGUF Q4_K_M** es el formato estándar de facto para inferencia en CPU/edge (llama.cpp, ~91K estrellas en GitHub), con ~5.35% de aumento de perplejidad típico vs fp16.
- **ExecuTorch** (Meta) llegó a v1.0 en oct-2025 como runtime de producción para edge — el área está activa y madura, no es un nicho muerto.
- Fuentes: On-Device LLMs State of the Union 2026 (v-chandra.github.io/on-device-llms), Awesome-LLMs-on-device (GitHub), documentación de llama.cpp.

## Qué ya tenemos (cero cómputo GPU nuevo necesario)

- Checkpoints ya entrenados: `lowrank-field-adapters/results/ablations_baselines_*/` (S³ full y LoRA r=8, dos semillas).
- Los factores SVD (`svd_factors/`) y el código de fusión es trivial: `W' = W + U_k(mθ(Σk)-Σk)Vk^T`, confirmado en `src/adapters/svmo_linear.py` — exactamente la misma forma que un merge de LoRA.
- El pipeline de evaluación (7 tareas lm-eval-harness) ya existe y ya corrió sobre estos mismos checkpoints — se reutiliza tal cual, solo cambia qué pesos se le dan.
- Un Android disponible para la fase de despliegue móvil.

## Plan de experimentos (por niveles, del más seguro al más ambicioso)

### Nivel 1 — Núcleo del paper (CPU pura, sin riesgo de herramientas, ejecutable ya)

1. Fusionar SVMO en los pesos base → `W_S3`. Fusionar LoRA → `W_LoRA`. (Script nuevo, ~50 líneas, reusa `svmo_linear.py`.)
2. Cuantizar ambos con `bitsandbytes` (NF4/INT8) **en PyTorch puro**, no GGUF todavía — esto evita pelear con el grafo estático de llama.cpp y permite mantener NMF/STB (que son módulos de inferencia, no pesos estáticos) corriendo en fp16 al lado del backbone cuantizado, tal como ya soporta el código actual de `S3Block`.
3. Medir: Δppl pre/post-cuantización (script de evaluación ya existe), accuracy en las 7 tareas pre/post, y — el test más barato y directo de la hipótesis — kurtosis y ratio max/std por canal de $W_{S3}$ vs $W_{LoRA}$ **antes de cuantizar siquiera** (puro NumPy, sin inferencia, minutos de cómputo).
4. Repetir en 2-3 bit-widths (8-bit, 4-bit, y si aguanta, un esquema tipo 3-bit) para ver si la brecha crece o se mantiene.

### Nivel 2 — Despliegue móvil (el gancho de "poor compute", ambicioso pero acotado)

5. Exportar **solo el backbone cuantizado** a GGUF vía llama.cpp (esto es 100% estándar para un modelo con pesos fusionados — LoRA-merged-and-quantized es el caso de uso más común de llama.cpp, cero riesgo).
6. Para el lado S³: el backbone con SVMO fusionado se exporta igual de fácil (también son solo pesos estáticos ya fusionados). **Honestidad necesaria de entrada:** NMF y STB no son pesos estáticos — son módulos que corren en tiempo de inferencia (una ODE y una cross-attention). llama.cpp no tiene forma nativa de alojar eso. Para la demo móvil, la opción honesta es probar el backbone-con-SVMO-fusionado solo (sin NMF/STB) como *ablación explícita*, dejando "NMF/STB en el teléfono" como trabajo futuro, no como algo que se resuelve en este proyecto.
7. Correr ambos GGUF en el Android (Termux + llama.cpp, ruta bien documentada) y reportar tokens/seg, RAM pico, tiempo de carga — números reales en hardware real.

## Lo que este proyecto NO afirma (para no repetir errores del pasado)

- No promete que S³ "gane" — la hipótesis puede salir falsa, y eso se reporta igual.
- No se necesita ni se usa la GT 710 para nada; es irrelevante para este proyecto, no un obstáculo a resolver.
- El despliegue móvil de NMF/STB completos queda fuera de alcance explícitamente — no se simula ni se proyecta, se declara como brecha abierta.
- Nada de esto se presenta como comparación "S³ vs. LoRA" en el sentido competitivo — es un estudio de **una propiedad estructural** (cómo cuantiza cada tipo de actualización), con LoRA como el aditivo de referencia de la literatura, igual que en el paper de S³.

## Nota sobre `holographic-function-embedding` (revisado a pedido)

Se revisó ese proyecto. Su propia README dice, en el mismo párrafo: "Shannon entropy of trained weights prohibits 70:1 lossless reduction" y dos líneas después promete exactamente eso ("no quality loss", modelo de 70B representado con ~10⁶ parámetros), solo con otro nombre ("no lo llamamos compresión"). Eso es una contradicción de teoría de la información dentro de su propio texto, no un detalle a pulir — no hay matemática desarrollada (todo dice "to be developed"), no hay código real (solo esqueleto de carpetas vacías), y el propio README tiene una marca de "no leer de aquí para abajo" al final, como si ya se supiera que estaba descartado. Recomendación: no es rescatable como está. Si se quiere retomar la idea de "representación compacta de la función en vez de los pesos" algún día, habría que empezar de cero con una afirmación mucho más modesta (compresión con pérdida acotada, no "sin pérdida"), pero eso es un proyecto totalmente distinto a PRISM y no se mezcla con este.

## Estado

Idea formalizada, plan de experimentos definido, cero corridas hechas todavía. Siguiente paso: escribir el script de fusión SVMO/LoRA y el análisis de kurtosis/outlier-ratio (Nivel 1, paso 3) — es la parte más barata y ya sería evidencia a favor o en contra de la hipótesis central antes de cuantizar nada.
