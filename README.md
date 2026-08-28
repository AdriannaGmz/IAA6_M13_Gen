# Modelos generativos profundos

**Diplomado en Inteligencia Artificial Aplicada · DGTIC-UNAM**
Material de prácticas del módulo de modelos generativos profundos.
Instructora: D. Adriana Gómez Rosal.

## Cómo empezar

1. Haga clic en el botón **Abrir en Colab** del cuaderno que corresponda.
2. **Guarde una copia en su propia unidad** (si no lo hace, pierde su
   trabajo al cerrar la pestaña).
3. Ejecute la primera celda (setup). Imprime si tiene GPU o no.
   **Sin GPU todo funciona igual, sólo más lento.**

## Los cuadernos

| Sesión | Cuaderno | Colab |
|---|---|---|
| — | `bitacora_plantilla.ipynb` — su bitácora personal | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdriannaGmz/IAA6_M13_Gen/blob/main/bitacora_plantilla.ipynb) |
| S1-E1 | `s1_e1_el_molde.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdriannaGmz/IAA6_M13_Gen/blob/main/s1_e1_el_molde.ipynb) |
| S1-E2 | `s1_e2_mi_primer_generador.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdriannaGmz/IAA6_M13_Gen/blob/main/s1_e2_mi_primer_generador.ipynb) |
| S2-E1 | `s2_e1_el_juego.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdriannaGmz/IAA6_M13_Gen/blob/main/s2_e1_el_juego.ipynb) |
| S2-E2 | `s2_e2_control.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdriannaGmz/IAA6_M13_Gen/blob/main/s2_e2_control.ipynb) |
| S3-E1 | `s3_e1_difusion_2d.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdriannaGmz/IAA6_M13_Gen/blob/main/s3_e1_difusion_2d.ipynb) |
| S3-E2 | `s3_e2_control_fino.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdriannaGmz/IAA6_M13_Gen/blob/main/s3_e2_control_fino.ipynb) |
| S4-E1 | `s4_e1_adaptar.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdriannaGmz/IAA6_M13_Gen/blob/main/s4_e1_adaptar.ipynb) |
| S4-E2 | `s4_e2_el_juicio.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdriannaGmz/IAA6_M13_Gen/blob/main/s4_e2_el_juicio.ipynb) |

## Ejecución local (alternativa a Colab)

```bash
git clone https://github.com/AdriannaGmz/IAA6_M13_Gen
cd IAA6_M13_Gen
pip install -r requirements.txt
jupyter lab
```

## Estructura

```
├── bitacora_plantilla.ipynb   ← lo que cada quien copia el primer día
├── s?_e?_*.ipynb              ← los ocho ejercicios
├── src/                       ← capa común: datos, modelos, evaluar,
│                                 graficas, rescate
├── assets/demo/               ← conjuntos de demostración (sintéticos,
│                                 con desbalance deliberado)
└── assets/checkpoints/        ← pesos precomputados (celdas de rescate)
```

## Los datos de demostración

Tres conjuntos **sintéticos** (generados por `scripts/preparar_conjuntos.py`,
semilla fija, sin datos de terceros), cada uno con una clase minoritaria
al ~5 % — el desbalance es deliberado y es la materia prima del
mini-proyecto:

| Modo | Contenido | Clase minoritaria |
|---|---|---|
| `imagen` | 3 000 figuras geométricas 32×32 en escala de grises | `anillo` (5.3 %) |
| `tabular` | 4 000 lecturas de sensores de un equipo rotatorio | `falla` (5 %) |
| `serie` | 1 200 series de 96 pasos (ciclo diario de un sensor) | `anomalia` (5 %) |

## Si algo falla

Cada cuaderno tiene **celdas de rescate**: ejecútelas y continúe con
resultados precomputados. Nadie se queda atrás por un problema técnico.

## Licencia

MIT. Ver `LICENSE`.
