"""Figuras estandarizadas de los ocho ejercicios.

Todas las funciones devuelven la figura de matplotlib, lista para
guardarse con ``fig.savefig(...)`` o pegarse en la bitácora.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


def _a_tensor(X):
    if isinstance(X, np.ndarray):
        return torch.from_numpy(X).float()
    return X.detach().cpu().float()


def _pintar_imagen(eje, imagen):
    """Dibuja un tensor (C,H,W) o (H,W) en [-1,1] sobre un eje."""
    im = (_a_tensor(imagen).clamp(-1, 1) + 1) / 2
    if im.ndim == 3 and im.shape[0] == 1:
        eje.imshow(im[0], cmap="gray", vmin=0, vmax=1)
    elif im.ndim == 3:
        eje.imshow(im.permute(1, 2, 0))
    else:
        eje.imshow(im, cmap="gray", vmin=0, vmax=1)
    eje.axis("off")


# ──────────────────────────────────────────────────────────────

def rejilla(X, titulo="", n=16):
    """Rejilla cuadrada de hasta n imágenes de X (N,C,H,W) en [-1,1]."""
    X = _a_tensor(X)
    n = min(n, len(X))
    lado = int(np.ceil(np.sqrt(n)))
    fig, ejes = plt.subplots(lado, lado, figsize=(1.6 * lado, 1.6 * lado))
    ejes = np.atleast_1d(ejes).ravel()
    for k, eje in enumerate(ejes):
        eje.axis("off")
        if k < n:
            _pintar_imagen(eje, X[k])
    if titulo:
        fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()
    return fig


def interpolacion(modelo, z1, z2, pasos=10):
    """Decodifica la línea recta entre dos latentes z1 y z2.

    Si la salida es imagen, dibuja un renglón de imágenes; si es
    tabular o serie, superpone las curvas decodificadas con un
    degradado de color.
    """
    z1, z2 = _a_tensor(z1).flatten(), _a_tensor(z2).flatten()
    alfas = torch.linspace(0, 1, pasos)
    zs = torch.stack([(1 - a) * z1 + a * z2 for a in alfas])
    with torch.no_grad():
        X = _a_tensor(modelo.decodificar(zs))

    if X.ndim == 4:  # imágenes
        fig, ejes = plt.subplots(1, pasos, figsize=(1.5 * pasos, 1.8))
        for k in range(pasos):
            _pintar_imagen(ejes[k], X[k])
            ejes[k].set_title(f"{float(alfas[k]):.1f}", fontsize=7)
        fig.suptitle("Interpolación en el espacio latente", fontsize=11)
    else:
        fig, eje = plt.subplots(figsize=(9, 4))
        colores = plt.cm.viridis(np.linspace(0, 1, pasos))
        for k in range(pasos):
            v = X[k, :, 0] if X.ndim == 3 else X[k]
            eje.plot(v.numpy(), color=colores[k], alpha=0.8)
        eje.set_title("Interpolación en el espacio latente "
                      "(morado → amarillo)", fontsize=11)
    fig.tight_layout()
    return fig


def curva(historial, titulo=""):
    """Curvas de entrenamiento.

    historial: lista de valores, o dict {nombre: lista de valores}.
    """
    if not isinstance(historial, dict):
        historial = {"pérdida": list(historial)}
    fig, eje = plt.subplots(figsize=(7, 4))
    for nombre, valores in historial.items():
        eje.plot(valores, label=nombre, linewidth=1.8)
    eje.set_xlabel("época")
    eje.set_ylabel("valor")
    eje.legend()
    eje.grid(alpha=0.3)
    if titulo:
        eje.set_title(titulo, fontsize=12)
    fig.tight_layout()
    return fig


def comparar(X_a, X_b, etiquetas=("A", "B")):
    """Comparación lado a lado de dos lotes de muestras.

    imagen (4D) -> dos rejillas | tabular (2D) -> histogramas
    superpuestos | serie (3D) -> dos paneles de series.
    """
    X_a, X_b = _a_tensor(X_a), _a_tensor(X_b)

    if X_a.ndim == 4:
        n = min(16, len(X_a), len(X_b))
        lado = int(np.ceil(np.sqrt(n)))
        fig, ejes = plt.subplots(lado, 2 * lado + 1,
                                 figsize=(1.4 * (2 * lado + 1), 1.5 * lado))
        ejes = np.atleast_2d(ejes)
        for k in range(lado * lado):
            f, c = divmod(k, lado)
            ejes[f][c].axis("off")
            ejes[f][lado].axis("off")           # columna separadora
            ejes[f][lado + 1 + c].axis("off")
            if k < n:
                _pintar_imagen(ejes[f][c], X_a[k])
                _pintar_imagen(ejes[f][lado + 1 + c], X_b[k])
        ejes[0][lado // 2].set_title(etiquetas[0], fontsize=12)
        ejes[0][lado + 1 + lado // 2].set_title(etiquetas[1], fontsize=12)

    elif X_a.ndim == 2:
        d = min(4, X_a.shape[1])
        fig, ejes = plt.subplots(1, d, figsize=(3.2 * d, 3))
        ejes = np.atleast_1d(ejes)
        for j in range(d):
            ejes[j].hist(X_a[:, j].numpy(), bins=30, alpha=0.6,
                         label=etiquetas[0], density=True)
            ejes[j].hist(X_b[:, j].numpy(), bins=30, alpha=0.6,
                         label=etiquetas[1], density=True)
            ejes[j].set_title(f"variable {j}", fontsize=9)
            ejes[j].tick_params(labelsize=7)
        ejes[0].legend(fontsize=8)

    else:  # series (N, L, C)
        n = min(8, len(X_a), len(X_b))
        fig, (e1, e2) = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
        for k in range(n):
            e1.plot(X_a[k, :, 0].numpy(), alpha=0.6)
            e2.plot(X_b[k, :, 0].numpy(), alpha=0.6)
        e1.set_title(etiquetas[0], fontsize=11)
        e2.set_title(etiquetas[1], fontsize=11)

    fig.tight_layout()
    return fig


def rejilla_barrido(resultados, eje_x, eje_y):
    """Rejilla 2D de un barrido cruzado de hiperparámetros.

    resultados: dict {(valor_x, valor_y): imagen} donde imagen es un
    tensor (C,H,W) o (H,W) en [-1,1].
    eje_x, eje_y: nombres de los dos hiperparámetros (para las etiquetas).
    """
    valores_x = sorted({k[0] for k in resultados})
    valores_y = sorted({k[1] for k in resultados})
    nx, ny = len(valores_x), len(valores_y)
    fig, ejes = plt.subplots(ny, nx, figsize=(1.9 * nx, 1.9 * ny))
    ejes = np.atleast_2d(ejes)
    for i, vy in enumerate(valores_y):
        for j, vx in enumerate(valores_x):
            eje = ejes[i][j]
            eje.axis("off")
            if (vx, vy) in resultados:
                _pintar_imagen(eje, resultados[(vx, vy)])
            if i == 0:
                eje.set_title(f"{eje_x}={vx}", fontsize=8)
            if j == 0:
                eje.axis("on")
                eje.set_xticks([])
                eje.set_yticks([])
                eje.set_ylabel(f"{eje_y}={vy}", fontsize=8)
                for lado in eje.spines.values():
                    lado.set_visible(False)
    fig.suptitle(f"Barrido {eje_x} × {eje_y}", fontsize=12)
    fig.tight_layout()
    return fig


def trayectoria_2d(pasos, titulos=None):
    """La secuencia visual del muestreo inverso de difusión en 2D.

    pasos: lista de arreglos (N, 2) — la nube de puntos en instantes
    sucesivos del proceso inverso, del ruido puro a la distribución.
    titulos: lista opcional de rótulos (por defecto "paso k").
    """
    k = len(pasos)
    fig, ejes = plt.subplots(1, k, figsize=(2.4 * k, 2.6),
                             sharex=True, sharey=True)
    ejes = np.atleast_1d(ejes)
    for i, nube in enumerate(pasos):
        nube = _a_tensor(nube).numpy()
        ejes[i].scatter(nube[:, 0], nube[:, 1], s=3, alpha=0.5,
                        color=plt.cm.plasma(i / max(k - 1, 1)))
        rotulo = titulos[i] if titulos else f"paso {i}"
        ejes[i].set_title(rotulo, fontsize=9)
        ejes[i].set_xticks([])
        ejes[i].set_yticks([])
        ejes[i].set_aspect("equal")
    fig.suptitle("Del ruido a la distribución — proceso inverso", fontsize=12)
    fig.tight_layout()
    return fig
