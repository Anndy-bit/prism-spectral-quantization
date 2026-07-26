# Resultados — Nivel 1, paso 2: cuantización real + perplejidad

Corrida completa (28/28 capas, streaming layer-major, sin GPU), held-out
idéntico al resto del laboratorio (últimos 8 ejemplos de Alpaca train,
mismo template). Fake-quantization int8 simétrica por canal de salida.
Script: `scripts/quantize_and_measure_ppl.py`.

## Resultado

| Variante | ppl fp32 | ppl int8 | Δppl |
|---|---:|---:|---:|
| SVMO (s3_full, seed 42) | 8.3904 | 8.3878 | **-0.0026** |
| LoRA (r=8, seed 42) | 5.7155 | 5.7300 | **+0.0145** |

## Lectura honesta

**Dirección consistente con la hipótesis, magnitud demasiado chica para
ser una afirmación fuerte.** LoRA degrada con la cuantización (+0.0145,
como se esperaría); SVMO no solo no degrada sino que el número baja
levemente (-0.0026) — probablemente ruido de redondeo a este nivel, no
una mejora real. El delta de LoRA es ~5.6× más grande que el de SVMO, en
la dirección que predice la hipótesis original (adaptación acotada
degrada menos). Pero ambos números son minúsculos en términos absolutos
(<0.3% de la perplejidad base de cada uno), sobre un set de solo 1021
tokens — a esta escala, no se puede afirmar que la diferencia sea
significativa y no ruido de medición.

**Nota aparte, no es el hallazgo de este experimento:** la perplejidad
base de LoRA (5.72) es mejor que la de SVMO (8.39) en este held-out —
eso es una comparación de calidad absoluta entre los dos checkpoints
(ya cubierta, con matices, en el paper de S³), no de robustez a la
cuantización. Lo que importa acá es el *delta* de cada uno con su propio
baseline, no comparar los dos valores absolutos entre sí.

## Por qué el efecto es tan chico: int8 es un cuantizador suave

8 bits por canal, simétrico, es una cuantización relativamente permisiva
— la mayoría de los métodos (LoRA incluido) la toleran casi sin pérdida.
Para que la diferencia estructural entre adaptación acotada y aditiva se
note de verdad, hace falta una cuantización más agresiva (4 bits), donde
el error de redondeo es mucho mayor y las colas/outliers de la
distribución de pesos (lo que mide el análisis de kurtosis del paso
anterior) importan más.

## Qué es publicable hoy

⚠️ **Direccionalmente a favor de SVMO, pero no concluyente:** a 8 bits,
ambos métodos son, en la práctica, indistinguibles de su versión sin
cuantizar. Hay una señal débil en la dirección esperada, no una prueba.

## Siguiente paso natural

Repetir exactamente el mismo pipeline (ya validado, estable, sin crashes)
con `QUANT_BITS = 4` en vez de 8 — cambio de una línea, mismo script,
mismo tiempo de corrida (~40 min por variante, ya medido). A 4 bits es
donde se espera que la diferencia entre acotado y aditivo se vuelva
grande y medible, no marginal.
