# Resultados — Nivel 1, paso 3: kurtosis/outlier-ratio por canal

Corrida completa: 196 matrices (28 capas × 7 tipos), checkpoints `s3_full` y
`lora_r8` de `ablations_baselines_20260719_193209` (misma fecha/semilla),
Qwen2.5-7B-Instruct. Script: `scripts/analyze_merge_outliers.py`.
Datos crudos: `results/outlier_stats.csv` (588 filas).

## Resultado principal (sin normalizar — "como se entrenó realmente")

| Variante | kurtosis media | max/std medio | \|ΔW\|/\|W\| medio |
|---|---:|---:|---:|
| base (sin adaptar) | 5.6745 | 4.8799 | — |
| + SVMO fusionado | 5.6745 | 4.8799 | 0.00051 |
| + LoRA fusionado | 5.5853 | 4.8776 | 0.02400 |

**SVMO deja las estadísticas por canal prácticamente intactas** (Δkurtosis
promedio = +0.000025, indistinguible de cero). **LoRA sí las mueve** de forma
medible (Δkurtosis promedio = −0.089, es decir, *reduce* la kurtosis en
promedio — aplana la cola, no la engorda como se hipotetizó originalmente).

En **190 de 196 matrices**, el cambio de kurtosis que produce LoRA es mayor
en magnitud que el que produce SVMO. Desglosado por tipo de matriz, la
diferencia es enorme y muy concentrada:

| Matriz | \|Δkurt\| SVMO | \|Δkurt\| LoRA | LoRA / SVMO |
|---|---:|---:|---:|
| q_proj | 0.00013 | 0.00170 | 13× |
| k_proj | 0.00037 | 0.00264 | 7× |
| v_proj | 0.00025 | 0.00117 | 5× |
| o_proj | 0.00031 | 0.00121 | 4× |
| **gate_proj** | 0.00054 | **0.5996** | **1104×** |
| **up_proj** | 0.00004 | 0.0152 | **390×** |
| down_proj | 0.00020 | 0.00370 | 19× |

El efecto se concentra brutalmente en las matrices MLP grandes (gate/up),
que son también las que más parámetros aportan al modelo.

## El matiz honesto que hay que reportar (no esconder)

El delta de LoRA es, en promedio, **47× más grande en norma de Frobenius
relativa** que el delta de SVMO (2.40% vs 0.051% de \|W\|). Es decir, parte
de "por qué LoRA mueve más las estadísticas" puede ser simplemente que **el
update de LoRA es mucho más grande en magnitud** en este par de checkpoints
concretos — no necesariamente que el *tipo* de update (aditivo sin cota vs.
espectral acotado) sea intrínsecamente más disruptivo por sí solo.

Se corrió también la versión normalizada (Δestadística por unidad de
\|ΔW\|/\|W\|) para intentar separar ambos efectos:

| Métrica normalizada | SVMO (media / mediana) | LoRA (media / mediana) |
|---|---:|---:|
| \|Δkurtosis\| / (\|ΔW\|/\|W\|) | 0.67 / 0.24 | 3.30 / 0.05 |
| \|Δ(max/std)\| / (\|ΔW\|/\|W\|) | 0.47 / 0.32 | 0.10 / 0.03 |

Este resultado normalizado **no es limpio**: la media y la mediana no
concuerdan entre sí (la media de LoRA está dominada por unas pocas matrices,
casi seguro gate/up_proj), y las dos métricas (kurtosis vs. max/std) apuntan
en direcciones distintas sobre cuál método es "más suave" por unidad de
cambio. **Conclusión honesta:** el resultado crudo (SVMO no mueve nada,
LoRA sí, sobre todo en gate/up) es sólido y reproducible. Pero todavía **no
se puede afirmar** que sea por el *tipo* de adaptación y no por el *tamaño*
del update — eso requeriría un experimento controlado (escalar
artificialmente el delta de LoRA para que tenga la misma norma de Frobenius
que el de SVMO, capa por capa, y remedir) que no se ha hecho todavía.

## Qué es publicable hoy sin forzar nada

✅ **Sostenible con datos medidos:**
- SVMO, tal como se entrena en la práctica, deja las estadísticas de peso
  por canal esencialmente sin cambios (Δkurtosis ≈ 0).
- LoRA, tal como se entrena en la práctica, sí las mueve, concentrado
  fuertemente en gate/up_proj (>390-1100× el efecto de SVMO ahí).
- Esta es una comparación "as-trained" realista — así es como se usan
  estos métodos en la práctica, nadie iguala manualmente la magnitud del
  update entre LoRA y SVMO hoy en día.

⚠️ **No sostenible todavía:**
- Que la causa sea el *tipo* de adaptación en sí (espectral-acotado vs.
  aditivo-sin-cota) independientemente de la magnitud — el resultado
  normalizado es mixto, no lo confirma limpiamente.
- Cualquier conclusión sobre qué tan bien cuantiza cada uno — esto todavía
  no mide cuantización, solo el efecto sobre las estadísticas de peso que
  los cuantizadores per-canal usan para calibrar escala.

## Próximo paso natural

Dos caminos, no mutuamente excluyentes:
1. **Seguir con el plan original** (Nivel 1, paso 2): cuantizar de verdad
   ambos merges (bitsandbytes NF4/INT8) y medir Δppl real — la pregunta que
   de verdad importa para el paper, y donde este resultado de kurtosis
   sirve como *motivación/explicación mecanística*, no como la afirmación
   central.
2. **Cerrar el matiz primero**: correr la versión con magnitud igualada
   (escalar el delta de LoRA a la misma \|ΔW\|/\|W\| que SVMO, capa por
   capa) para saber si el efecto sobrevive al controlar por tamaño. Es
   barato (~mismo script, unas líneas más) y evita que un reviewer haga la
   misma pregunta que nos hicimos nosotros mismos aquí.
