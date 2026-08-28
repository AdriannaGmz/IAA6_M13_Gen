"""Genera los tres conjuntos de demostración con desbalance deliberado.

Los tres son SINTÉTICOS y deterministas (semilla fija): sin problemas de
licencia, sin descargas externas, reproducibles al byte. El desbalance
(~5 % en una clase) es intencional: la clase minoritaria del
mini-proyecto existe desde el primer día (Plan de prácticas §D.2.2).

Requisitos que verifica este script al final:
- imagen  < 10 MB · tabular < 2 MB · serie < 2 MB
- cada conjunto se carga en < 10 segundos

Uso:  python scripts/preparar_conjuntos.py
"""

from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
DEMO = RAIZ / "assets" / "demo"
SEMILLA = 20260828  # fecha de inicio del módulo

rng = np.random.default_rng(SEMILLA)


# ══════════════════════════════════════════════════════════════
# 1. IMAGEN — figuras geométricas 32×32, clase "anillo" al ~5 %
# ══════════════════════════════════════════════════════════════

def _lienzo():
    return np.zeros((32, 32), dtype=np.float32)


def _coordenadas():
    yy, xx = np.mgrid[0:32, 0:32]
    return xx.astype(np.float32), yy.astype(np.float32)


def dibujar_circulo(cx, cy, r):
    xx, yy = _coordenadas()
    im = _lienzo()
    im[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] = 1.0
    return im


def dibujar_anillo(cx, cy, r, grosor):
    xx, yy = _coordenadas()
    im = _lienzo()
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    im[(d2 <= r ** 2) & (d2 >= (r - grosor) ** 2)] = 1.0
    return im


def dibujar_cuadrado(cx, cy, medio_lado):
    im = _lienzo()
    x0, x1 = int(cx - medio_lado), int(cx + medio_lado)
    y0, y1 = int(cy - medio_lado), int(cy + medio_lado)
    im[max(y0, 0):y1, max(x0, 0):x1] = 1.0
    return im


def dibujar_triangulo(cx, cy, medio_lado):
    xx, yy = _coordenadas()
    im = _lienzo()
    altura = medio_lado * 2
    dentro = (
        (yy >= cy - medio_lado)
        & (yy <= cy + medio_lado)
        & (np.abs(xx - cx) <= (yy - (cy - medio_lado)) / altura * medio_lado)
    )
    im[dentro] = 1.0
    return im


def dibujar_cruz(cx, cy, medio_lado, grosor):
    im = _lienzo()
    x0, x1 = int(cx - medio_lado), int(cx + medio_lado)
    y0, y1 = int(cy - medio_lado), int(cy + medio_lado)
    g = int(grosor)
    im[max(int(cy) - g, 0):int(cy) + g, max(x0, 0):x1] = 1.0
    im[max(y0, 0):y1, max(int(cx) - g, 0):int(cx) + g] = 1.0
    return im


def generar_imagenes():
    # anillo es la clase minoritaria: ~5 % del total
    plan = [
        ("circulo", 710),
        ("cuadrado", 710),
        ("triangulo", 710),
        ("cruz", 710),
        ("anillo", 160),
    ]
    imagenes, etiquetas = [], []
    for idx_clase, (nombre, n) in enumerate(plan):
        for _ in range(n):
            cx = rng.uniform(12, 20)
            cy = rng.uniform(12, 20)
            r = rng.uniform(6, 11)
            if nombre == "circulo":
                im = dibujar_circulo(cx, cy, r)
            elif nombre == "cuadrado":
                im = dibujar_cuadrado(cx, cy, r * 0.85)
            elif nombre == "triangulo":
                im = dibujar_triangulo(cx, cy, r)
            elif nombre == "cruz":
                im = dibujar_cruz(cx, cy, r, grosor=rng.uniform(1.5, 3))
            else:
                im = dibujar_anillo(cx, cy, r, grosor=rng.uniform(2, 4))
            # intensidad variable + ruido de fondo suave
            im = im * rng.uniform(0.7, 1.0)
            im = im + rng.normal(0, 0.04, im.shape).astype(np.float32)
            imagenes.append(np.clip(im, 0, 1))
            etiquetas.append(idx_clase)

    X = (np.stack(imagenes) * 255).astype(np.uint8)
    y = np.array(etiquetas, dtype=np.int64)
    orden = rng.permutation(len(y))
    X, y = X[orden], y[orden]

    destino = DEMO / "imagen" / "demo_imagen.npz"
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destino, X=X, y=y,
        nombres_clases=np.array([p[0] for p in plan]),
    )
    return destino, y, [p[0] for p in plan]


# ══════════════════════════════════════════════════════════════
# 2. TABULAR — fallas de equipo rotatorio, clase "falla" al ~5 %
# ══════════════════════════════════════════════════════════════

def generar_tabular():
    n_normal, n_falla = 3800, 200
    columnas = ["temperatura_C", "vibracion_mm_s", "presion_bar",
                "corriente_A", "horas_desde_mantenimiento", "rpm"]

    # Operación normal: valores correlacionados alrededor del punto nominal
    temp = rng.normal(62, 5, n_normal)
    vib = rng.normal(2.2, 0.5, n_normal) + 0.03 * (temp - 62)
    pres = rng.normal(6.0, 0.5, n_normal)
    corr = rng.normal(14, 1.5, n_normal) + 0.05 * (temp - 62)
    horas = rng.uniform(0, 2000, n_normal)
    rpm = rng.normal(1750, 60, n_normal)
    X_normal = np.stack([temp, vib, pres, corr, horas, rpm], axis=1)

    # Fallas: dos submodos (sobrecalentamiento y desgaste de rodamiento)
    n_a = n_falla // 2
    n_b = n_falla - n_a
    # submodo A — sobrecalentamiento: temperatura y corriente altas
    Xa = np.stack([
        rng.normal(88, 6, n_a),
        rng.normal(3.5, 0.8, n_a),
        rng.normal(5.2, 0.6, n_a),
        rng.normal(19, 2.0, n_a),
        rng.uniform(800, 2400, n_a),
        rng.normal(1710, 80, n_a),
    ], axis=1)
    # submodo B — rodamiento: vibración alta, rpm inestable, muchas horas
    Xb = np.stack([
        rng.normal(70, 6, n_b),
        rng.normal(6.5, 1.2, n_b),
        rng.normal(5.8, 0.5, n_b),
        rng.normal(15, 1.8, n_b),
        rng.uniform(1500, 2600, n_b),
        rng.normal(1650, 140, n_b),
    ], axis=1)

    X = np.concatenate([X_normal, Xa, Xb]).astype(np.float32)
    y = np.concatenate([
        np.zeros(n_normal, dtype=np.int64),
        np.ones(n_falla, dtype=np.int64),
    ])
    orden = rng.permutation(len(y))
    X, y = X[orden], y[orden]

    destino = DEMO / "tabular" / "demo_tabular.npz"
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destino, X=X, y=y,
        nombres_clases=np.array(["operacion_normal", "falla"]),
        nombres_columnas=np.array(columnas),
    )

    # copia CSV legible para inspección humana
    import csv
    ruta_csv = DEMO / "tabular" / "demo_tabular.csv"
    with open(ruta_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columnas + ["etiqueta"])
        for fila, et in zip(X, y):
            w.writerow([f"{v:.3f}" for v in fila] + [int(et)])

    return destino, y, ["operacion_normal", "falla"]


# ══════════════════════════════════════════════════════════════
# 3. SERIE — sensor con ciclo diario, anomalías raras (~5 %)
# ══════════════════════════════════════════════════════════════

def generar_series():
    n_normal, n_anomalia = 1140, 60
    L = 96  # una lectura cada 15 min durante 24 h
    t = np.linspace(0, 2 * np.pi, L)

    series, etiquetas = [], []
    for _ in range(n_normal):
        base = (
            rng.uniform(0.8, 1.2) * np.sin(t + rng.uniform(-0.4, 0.4))
            + rng.uniform(-0.2, 0.2)
            + rng.normal(0, 0.08, L)
        )
        series.append(base)
        etiquetas.append(0)

    for _ in range(n_anomalia):
        base = (
            rng.uniform(0.8, 1.2) * np.sin(t + rng.uniform(-0.4, 0.4))
            + rng.uniform(-0.2, 0.2)
            + rng.normal(0, 0.08, L)
        )
        tipo = rng.integers(0, 3)
        if tipo == 0:      # pico abrupto
            pos = rng.integers(10, L - 10)
            base[pos:pos + 3] += rng.uniform(2.5, 4.0)
        elif tipo == 1:    # escalón sostenido
            pos = rng.integers(20, L - 30)
            base[pos:] += rng.uniform(1.2, 2.0)
        else:              # ráfaga de oscilación
            pos = rng.integers(10, L - 25)
            base[pos:pos + 20] += 0.9 * np.sin(np.linspace(0, 12 * np.pi, 20))
        series.append(base)
        etiquetas.append(1)

    X = np.stack(series).astype(np.float32)
    y = np.array(etiquetas, dtype=np.int64)
    orden = rng.permutation(len(y))
    X, y = X[orden], y[orden]

    destino = DEMO / "serie" / "demo_serie.npz"
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destino, X=X, y=y,
        nombres_clases=np.array(["normal", "anomalia"]),
    )
    return destino, y, ["normal", "anomalia"]


# ══════════════════════════════════════════════════════════════

def _reporte(nombre, ruta, y, nombres, limite_mb):
    mb = ruta.stat().st_size / 1e6
    conteo = np.bincount(y)
    detalle = ", ".join(
        f"{nombres[i]}={conteo[i]} ({conteo[i] / len(y):.1%})"
        for i in range(len(conteo))
    )
    estado = "OK" if mb < limite_mb else f"EXCEDE el límite de {limite_mb} MB"
    print(f"[{nombre}] {ruta.name}: {mb:.2f} MB ({estado}) · N={len(y)} · {detalle}")
    assert mb < limite_mb, f"{nombre} excede el límite de tamaño"


if __name__ == "__main__":
    r1, y1, n1 = generar_imagenes()
    r2, y2, n2 = generar_tabular()
    r3, y3, n3 = generar_series()
    print()
    _reporte("imagen ", r1, y1, n1, limite_mb=10)
    _reporte("tabular", r2, y2, n2, limite_mb=2)
    _reporte("serie  ", r3, y3, n3, limite_mb=2)
    print("\nConjuntos de demostración listos.")
