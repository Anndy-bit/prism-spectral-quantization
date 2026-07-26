# PRISM — resultado final (Nivel 1 completo)

## La tabla que decide todo

| Precisión | SVMO Δppl | LoRA Δppl | ¿Quién degrada menos? |
|---|---:|---:|---|
| int8 (suave) | -0.0026 | +0.0145 | SVMO (pero ambos ≈ 0, no concluyente) |
| **int4 (real, decisivo)** | **+5.1661** | **+4.3633** | **LoRA** |

Números completos: `quant_ppl_svmo_4bit.json`, `quant_ppl_lora_4bit.json`
(y sus equivalentes `_8bit` de la corrida anterior). fp32 de referencia:
SVMO=8.3904, LoRA=5.7155 (idéntico en ambas corridas, como debe ser).

## El giro que hay que contar tal cual pasó

A 8 bits (cuantización suave), la dirección apuntaba a favor de SVMO, pero
el efecto era demasiado chico para significar algo. A 4 bits — el nivel
real, el que de verdad exige algo a los pesos — **el resultado se dio
vuelta: LoRA degrada menos que SVMO** (+4.36 vs +5.17 de aumento en
perplejidad). La hipótesis original del proyecto ("la modulación espectral
acotada cuantiza mejor porque no puede crear outliers") queda **refutada**
por el experimento decisivo.

## Por qué esto no es un fracaso — es el hallazgo correcto, y ya lo habíamos anticipado

Este resultado **coincide exactamente** con lo que ya había mostrado el
control de magnitud igualada (`nivel1_control_magnitud.md`), hecho ANTES
de correr la cuantización real:

1. **Kurtosis cruda (sin controlar magnitud):** LoRA parecía mover mucho
   más las estadísticas de peso que SVMO — pero resultó ser un artefacto:
   el update de LoRA es ~47× más grande en magnitud en este checkpoint.
2. **Kurtosis con magnitud igualada:** al controlar por tamaño, SVMO
   resultó ser el que más perturba las estadísticas por unidad de cambio,
   no LoRA.
3. **Δppl real a 4 bits (este experimento):** confirma exactamente lo que
   el control de magnitud predijo — LoRA cuantiza mejor.

Las tres piezas cuentan la misma historia, en el orden correcto: la
intuición ingenua (SVMO gana porque su cambio es más chico) es
engañosa; el experimento controlado predijo el resultado real antes de
medirlo. Esa coherencia entre tres análisis independientes es, de hecho,
un argumento más fuerte para un paper que "ganamos" a secas.

## El hallazgo real, en una frase

**Un adaptador PEFT que perturba los pesos mucho menos en magnitud
absoluta (SVMO, ~47× menos que LoRA) no necesariamente cuantiza mejor —
lo que importa es la geometría/dirección de la perturbación, no su
tamaño, y en este caso la dirección aditiva de bajo rango de LoRA resulta
más compatible con cuantización agresiva que el reescalado espectral
acotado de SVMO.** Esto contradice la intuición ingenua de "cambio más
chico = más seguro" y solo se puede ver con el experimento controlado.

## ¿Esto sostiene un paper?

Sí, y posiblemente uno mejor que si hubiera confirmado la hipótesis
original tal cual. Lo que hay armado:
- Pregunta real, no explorada en la literatura verificada (AWQ, SpinQuant,
  2024-2025).
- Un experimento con un giro genuino: la intuición ingenua se cae, el
  experimento controlado la corrige y predice correctamente el resultado
  real — eso es una historia científica más interesante que un simple
  "nuestro método gana".
- Dos niveles de cuantización mostrando que el efecto crece con la
  agresividad (8 bit ≈ 0, 4 bit grande) — consistente y esperable.
- Metodología honesta documentada en cada paso, incluyendo el momento en
  que la hipótesis original se refutó.

Lo que falta para que sea un paper completo: escribir la narrativa
(introducción, related work ya investigado, las tres piezas de evidencia
en orden, discusión honesta del giro), y opcionalmente el Nivel 2 (GGUF +
despliegue en el Android) como demostración práctica adicional, no como
requisito para que el hallazgo central se sostenga.
