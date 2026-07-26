# Control de magnitud igualada — resultado

Escalé el delta de LoRA, capa por capa, para que tenga la misma norma de
Frobenius que el delta de SVMO en esa misma matriz, y volví a medir kurtosis
y max/std. Verificación: la diferencia de magnitud entre ambos tras el
escalado es ~1e-6 (el escalado funcionó).

## Resultado: la hipótesis original NO sobrevive el control

| | \|Δkurtosis\| media | \|Δkurtosis\| mediana |
|---|---:|---:|
| SVMO | 0.00026 | 0.00008 |
| LoRA crudo (como se entrenó) | 0.08932 | 0.00123 |
| **LoRA con magnitud = SVMO** | **0.00004** | **0.00000** |

Una vez igualada la magnitud, LoRA deja de ser "más disruptivo" — pasa a ser
**igual o menos disruptivo que SVMO** en 188 de 196 matrices (kurtosis) y en
179 de 196 (max/std). Por tipo de matriz, en gate/up_proj (donde antes LoRA
"ganaba" por 390-1100×), con magnitud igualada SVMO es 3× más disruptivo,
no al revés.

## Veredicto honesto

**La hipótesis tal como estaba escrita en el README de PRISM ("la geometría
espectral acotada es intrínsecamente más suave para cuantización,
independientemente de su magnitud") queda refutada por este control.** El
efecto original era, casi enteramente, un artefacto de que el update de
LoRA es ~47× más grande en magnitud en este par de checkpoints — no una
propiedad estructural de "aditivo vs. espectral" en sí misma. Bien que lo
comprobamos antes de escribirlo como si fuera un hecho.

## Pero esto no mata el proyecto — lo redefine

Lo que sí es real, medido, y sigue siendo interesante: **SVMO alcanza
precisión downstream comparable a LoRA r=8 (ya reportado en el paper de S³,
Tabla III) perturbando los pesos ~47× menos en norma de Frobenius.** No es
que la forma del update importe per se — es que SVMO logra el mismo
resultado con un cambio absoluto mucho más chico. Esa es la afirmación
correcta, y sigue siendo una pregunta legítima si eso se traduce en menor
degradación al cuantizar.

La comparación "as-trained" (sin igualar magnitud) del paso anterior sigue
siendo válida como pregunta práctica: nadie despliega una versión de LoRA
artificialmente achicada — se despliega la que realmente se entrenó. Así
que medir Δppl real de ambos modelos, tal como existen hoy, tras
cuantizarlos, sigue siendo el experimento que decide si esto es un paper.

## Siguiente paso

Cuantización real (Nivel 1, paso 2, ya reformulado): merge + bitsandbytes
NF4/INT8 + Δppl sobre el held-out de Alpaca, para ambos checkpoints tal
como fueron entrenados. Este resultado de hoy es el que explica el *porqué*
si el Δppl sale distinto — ya no "por la forma", sino "por la magnitud
necesaria para llegar al mismo lugar".
