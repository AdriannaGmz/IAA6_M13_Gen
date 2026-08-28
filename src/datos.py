"""Carga, partición e inspección de datos para las tres modalidades.

Contrato congelado (Plan de prácticas §A.2). La única línea que cambia
entre un participante con imágenes y uno con una tabla es:

    X, y, meta = datos.cargar("tabular", "mis_datos.csv")

Convenciones de archivos propios:
- imagen : carpeta con una subcarpeta por clase (PNG/JPG), o un .npz
           con claves ``X`` (N,H,W) o (N,H,W,C) y ``y`` (N,).
- tabular: un .csv cuyas columnas son numéricas; la etiqueta se toma de
           la columna llamada ``etiqueta`` si existe, o de la última
           columna en caso contrario. También acepta .npz con ``X``, ``y``.
- serie  : un .npz con claves ``X`` (N,L) o (N,L,C) y ``y`` (N,), o un
           .csv donde cada renglón es una serie y la última columna
           (o la columna ``etiqueta``) es la etiqueta.
"""

from pathlib import Path

import numpy as np
import torch

_RUTA_DEMO = Path(__file__).resolve().parent.parent / "assets" / "demo"

_ARCHIVOS_DEMO = {
    "imagen": _RUTA_DEMO / "imagen" / "demo_imagen.npz",
    "tabular": _RUTA_DEMO / "tabular" / "demo_tabular.npz",
    "serie": _RUTA_DEMO / "serie" / "demo_serie.npz",
}

MODOS_VALIDOS = ("imagen", "tabular", "serie")


# ──────────────────────────────────────────────────────────────
# Función principal
# ──────────────────────────────────────────────────────────────

def cargar(modo, fuente=None, tam=32):
    """Carga un conjunto de datos y lo deja listo para los modelos.

    Parámetros
    ----------
    modo   : "imagen" | "tabular" | "serie"
    fuente : None -> conjunto de demostración del módulo
             ruta -> archivo del participante (.csv, .npz, carpeta de imágenes)
    tam    : lado de la imagen (32 por defecto; 64 si hay GPU)

    Devuelve
    --------
    (X, y, meta)
      X    : torch.Tensor normalizado a [-1, 1]
             imagen  -> (N, C, H, W)
             tabular -> (N, D)
             serie   -> (N, L, C)
      y    : torch.Tensor de etiquetas enteras (N,)
      meta : dict con claves
             n_muestras, n_clases, nombres_clases, conteo_por_clase,
             clase_minoritaria, desbalance, modalidad, forma, fuente
             (más claves internas con prefijo "_" para desnormalizar)
    """
    if modo not in MODOS_VALIDOS:
        raise ValueError(f"modo debe ser uno de {MODOS_VALIDOS}, no {modo!r}")

    if fuente is None:
        ruta = _ARCHIVOS_DEMO[modo]
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró el conjunto de demostración {ruta}.\n"
                "Ejecute scripts/preparar_conjuntos.py o clone el "
                "repositorio completo."
            )
        etiqueta_fuente = "demostración"
    else:
        ruta = Path(fuente)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe la ruta {ruta}")
        etiqueta_fuente = str(fuente)

    if modo == "imagen":
        X, y, nombres, extra = _cargar_imagen(ruta, tam)
    elif modo == "tabular":
        X, y, nombres, extra = _cargar_tabular(ruta)
    else:
        X, y, nombres, extra = _cargar_serie(ruta)

    meta = _armar_meta(X, y, nombres, modo, etiqueta_fuente, extra)
    return X, y, meta


# ──────────────────────────────────────────────────────────────
# Cargadores por modalidad
# ──────────────────────────────────────────────────────────────

def _cargar_imagen(ruta, tam):
    if ruta.is_dir():
        X_np, y_np, nombres = _leer_carpeta_imagenes(ruta, tam)
    elif ruta.suffix == ".npz":
        d = np.load(ruta, allow_pickle=False)
        X_np = d["X"].astype(np.float32)
        y_np = d["y"].astype(np.int64)
        nombres = [str(s) for s in d["nombres_clases"]] if "nombres_clases" in d else None
    else:
        raise ValueError(
            "Para modo='imagen' la fuente debe ser una carpeta con "
            "subcarpetas por clase, o un .npz con claves X, y."
        )

    # A forma (N, C, H, W)
    if X_np.ndim == 3:                      # (N, H, W) -> gris
        X_np = X_np[:, None, :, :]
    elif X_np.ndim == 4 and X_np.shape[-1] in (1, 3):  # (N, H, W, C)
        X_np = X_np.transpose(0, 3, 1, 2)

    X = torch.from_numpy(np.ascontiguousarray(X_np)).float()

    # Rango [0,255] o [0,1] -> [-1, 1]
    maximo = float(X.max())
    if maximo > 1.5:
        X = X / 127.5 - 1.0
    else:
        X = X * 2.0 - 1.0
    X = X.clamp(-1.0, 1.0)

    if X.shape[-1] != tam:
        X = torch.nn.functional.interpolate(
            X, size=(tam, tam), mode="bilinear", align_corners=False
        )

    y = torch.from_numpy(y_np).long()
    return X, y, nombres, {}


def _leer_carpeta_imagenes(carpeta, tam):
    from PIL import Image  # PIL es dependencia de matplotlib/torchvision

    subcarpetas = sorted(p for p in carpeta.iterdir() if p.is_dir())
    if not subcarpetas:
        raise ValueError(
            f"La carpeta {carpeta} no tiene subcarpetas por clase. "
            "Estructura esperada: carpeta/clase_a/*.png, carpeta/clase_b/*.png"
        )
    imagenes, etiquetas, nombres = [], [], []
    for idx, sub in enumerate(subcarpetas):
        nombres.append(sub.name)
        archivos = sorted(
            f for f in sub.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
        )
        for f in archivos:
            im = Image.open(f).convert("L").resize((tam, tam))
            imagenes.append(np.asarray(im, dtype=np.float32))
            etiquetas.append(idx)
    if not imagenes:
        raise ValueError(f"No se encontraron imágenes en {carpeta}")
    return np.stack(imagenes), np.array(etiquetas, dtype=np.int64), nombres


def _cargar_tabular(ruta):
    nombres_columnas = None
    if ruta.suffix == ".npz":
        d = np.load(ruta, allow_pickle=False)
        X_np = d["X"].astype(np.float32)
        y_np = d["y"].astype(np.int64)
        nombres = [str(s) for s in d["nombres_clases"]] if "nombres_clases" in d else None
        if "nombres_columnas" in d:
            nombres_columnas = [str(s) for s in d["nombres_columnas"]]
    elif ruta.suffix == ".csv":
        X_np, y_np, nombres, nombres_columnas = _leer_csv(ruta)
    else:
        raise ValueError("Para modo='tabular' la fuente debe ser .csv o .npz")

    X = torch.from_numpy(X_np).float()
    X, norm = _normalizar_por_columna(X)
    y = torch.from_numpy(y_np).long()
    extra = {"_norm": norm}
    if nombres_columnas is not None:
        extra["_columnas"] = nombres_columnas
    return X, y, nombres, extra


def _leer_csv(ruta):
    import pandas as pd

    df = pd.read_csv(ruta)
    if "etiqueta" in df.columns:
        col_y = "etiqueta"
    else:
        col_y = df.columns[-1]
    serie_y = df[col_y]
    if serie_y.dtype == object:
        categorias = sorted(serie_y.astype(str).unique())
        nombres = list(categorias)
        y_np = serie_y.astype(str).map({c: i for i, c in enumerate(categorias)}).to_numpy()
    else:
        y_np = serie_y.to_numpy()
        nombres = None
    df_X = df.drop(columns=[col_y])
    no_numericas = [c for c in df_X.columns if not np.issubdtype(df_X[c].dtype, np.number)]
    if no_numericas:
        raise ValueError(
            f"Columnas no numéricas en el CSV: {no_numericas}. "
            "Conviértalas a números (o elimínelas) antes de cargar."
        )
    return (
        df_X.to_numpy(dtype=np.float32),
        y_np.astype(np.int64),
        nombres,
        list(df_X.columns),
    )


def _cargar_serie(ruta):
    if ruta.suffix == ".npz":
        d = np.load(ruta, allow_pickle=False)
        X_np = d["X"].astype(np.float32)
        y_np = d["y"].astype(np.int64)
        nombres = [str(s) for s in d["nombres_clases"]] if "nombres_clases" in d else None
    elif ruta.suffix == ".csv":
        X_tab, y_np, nombres, _ = _leer_csv(ruta)
        X_np = X_tab  # cada renglón es una serie
    else:
        raise ValueError("Para modo='serie' la fuente debe ser .npz o .csv")

    if X_np.ndim == 2:          # (N, L) -> (N, L, 1)
        X_np = X_np[:, :, None]

    X = torch.from_numpy(X_np).float()
    X, norm = _normalizar_por_canal(X)
    y = torch.from_numpy(y_np).long()
    return X, y, nombres, {"_norm": norm}


# ──────────────────────────────────────────────────────────────
# Normalización a [-1, 1]
# ──────────────────────────────────────────────────────────────

def _normalizar_por_columna(X):
    minimo = X.min(dim=0).values
    maximo = X.max(dim=0).values
    rango = (maximo - minimo).clamp(min=1e-8)
    X_norm = (X - minimo) / rango * 2.0 - 1.0
    return X_norm, {"min": minimo, "max": maximo}


def _normalizar_por_canal(X):
    # X: (N, L, C); mínimo y máximo por canal
    minimo = X.amin(dim=(0, 1))
    maximo = X.amax(dim=(0, 1))
    rango = (maximo - minimo).clamp(min=1e-8)
    X_norm = (X - minimo) / rango * 2.0 - 1.0
    return X_norm, {"min": minimo, "max": maximo}


def desnormalizar(X, meta):
    """Regresa X del rango [-1, 1] a las unidades originales.

    Para imagen devuelve valores en [0, 255]; para tabular y serie usa
    los mínimos y máximos guardados al cargar.
    """
    if meta["modalidad"] == "imagen":
        return (X.clamp(-1, 1) + 1.0) * 127.5
    norm = meta.get("_norm")
    if norm is None:
        return X
    rango = (norm["max"] - norm["min"]).clamp(min=1e-8)
    return (X.clamp(-1, 1) + 1.0) / 2.0 * rango + norm["min"]


# ──────────────────────────────────────────────────────────────
# Meta, partición, ficha y vistazo
# ──────────────────────────────────────────────────────────────

def _armar_meta(X, y, nombres, modo, fuente, extra):
    clases, conteos = torch.unique(y, return_counts=True)
    n_clases = len(clases)
    if nombres is None:
        nombres = [f"clase_{int(c)}" for c in clases]
    conteo_por_clase = {
        nombres[i]: int(conteos[i]) for i in range(n_clases)
    }
    idx_min = int(torch.argmin(conteos))
    meta = {
        "n_muestras": int(X.shape[0]),
        "n_clases": n_clases,
        "nombres_clases": list(nombres),
        "conteo_por_clase": conteo_por_clase,
        "clase_minoritaria": nombres[idx_min],
        "desbalance": round(float(conteos[idx_min]) / float(X.shape[0]), 4),
        "modalidad": modo,
        "forma": tuple(X.shape),
        "fuente": fuente,
    }
    meta.update(extra)
    return meta


def dividir(X, y, prop_prueba=0.2, semilla=0):
    """Partición estratificada. Devuelve (X_ent, y_ent, X_pru, y_pru)."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    idx_ent, idx_pru = train_test_split(
        idx,
        test_size=prop_prueba,
        random_state=semilla,
        stratify=y.numpy(),
    )
    return X[idx_ent], y[idx_ent], X[idx_pru], y[idx_pru]


def ficha(meta):
    """Imprime y devuelve la ficha de datos formateada para la bitácora."""
    renglones = [
        "┌─ FICHA DE DATOS ────────────────────────────────",
        f"│ Modalidad          : {meta['modalidad']}",
        f"│ Fuente             : {meta['fuente']}",
        f"│ Forma del tensor X : {meta['forma']}",
        f"│ Número de muestras : {meta['n_muestras']}",
        f"│ Número de clases   : {meta['n_clases']}",
        "│ Conteo por clase   :",
    ]
    total = meta["n_muestras"]
    for nombre, n in meta["conteo_por_clase"].items():
        marca = "  ← minoritaria" if nombre == meta["clase_minoritaria"] else ""
        renglones.append(f"│   · {nombre:<22s} {n:>6d}  ({n / total:6.1%}){marca}")
    renglones += [
        f"│ Clase minoritaria  : {meta['clase_minoritaria']} "
        f"({meta['desbalance']:.1%} del total)",
        "└─────────────────────────────────────────────────",
    ]
    texto = "\n".join(renglones)
    print(texto)
    return texto


def vistazo(X, y, meta, n=16):
    """Figura de muestra apropiada a la modalidad.

    imagen -> rejilla | tabular -> pares de dispersión | serie -> superposición.
    Devuelve la figura de matplotlib.
    """
    import matplotlib.pyplot as plt

    modo = meta["modalidad"]
    nombres = meta["nombres_clases"]

    if modo == "imagen":
        n = min(n, len(X))
        # muestra balanceada: intenta incluir todas las clases
        indices = _indices_representativos(y, n)
        lado = int(np.ceil(np.sqrt(n)))
        fig, ejes = plt.subplots(lado, lado, figsize=(1.6 * lado, 1.8 * lado))
        ejes = np.atleast_1d(ejes).ravel()
        for k, eje in enumerate(ejes):
            eje.axis("off")
            if k < n:
                i = indices[k]
                im = (X[i].clamp(-1, 1) + 1) / 2
                if im.shape[0] == 1:
                    eje.imshow(im[0], cmap="gray", vmin=0, vmax=1)
                else:
                    eje.imshow(im.permute(1, 2, 0))
                eje.set_title(nombres[int(y[i])], fontsize=8)
        fig.suptitle(f"Vistazo · {meta['n_muestras']} imágenes", fontsize=11)

    elif modo == "tabular":
        columnas = meta.get("_columnas") or [f"x{j}" for j in range(X.shape[1])]
        d = min(4, X.shape[1])
        fig, ejes = plt.subplots(d, d, figsize=(2.2 * d, 2.2 * d))
        ejes = np.atleast_2d(ejes)
        colores = plt.cm.tab10(y.numpy() % 10)
        for i in range(d):
            for j in range(d):
                eje = ejes[i][j]
                if i == j:
                    for c in range(meta["n_clases"]):
                        eje.hist(X[y == c, i].numpy(), bins=25, alpha=0.6,
                                 label=nombres[c])
                else:
                    eje.scatter(X[:, j].numpy(), X[:, i].numpy(),
                                s=4, c=colores, alpha=0.5, linewidths=0)
                if i == d - 1:
                    eje.set_xlabel(columnas[j], fontsize=8)
                if j == 0:
                    eje.set_ylabel(columnas[i], fontsize=8)
                eje.tick_params(labelsize=6)
        ejes[0][0].legend(fontsize=7)
        fig.suptitle(
            f"Vistazo · pares de dispersión (primeras {d} variables)",
            fontsize=11,
        )

    else:  # serie
        n = min(n, len(X))
        indices = _indices_representativos(y, n)
        fig, eje = plt.subplots(figsize=(9, 4))
        ya_etiquetada = set()
        for i in indices:
            c = int(y[i])
            etiqueta = nombres[c] if c not in ya_etiquetada else None
            ya_etiquetada.add(c)
            eje.plot(X[i, :, 0].numpy(), alpha=0.7,
                     color=plt.cm.tab10(c % 10), label=etiqueta)
        eje.legend(fontsize=8)
        eje.set_xlabel("tiempo")
        eje.set_ylabel("valor normalizado")
        eje.set_title(f"Vistazo · {n} series superpuestas", fontsize=11)

    fig.tight_layout()
    return fig


def _indices_representativos(y, n):
    """Índices que intentan cubrir todas las clases antes de repetir."""
    indices = []
    clases = torch.unique(y).tolist()
    por_clase = {c: (y == c).nonzero(as_tuple=True)[0].tolist() for c in clases}
    ronda = 0
    while len(indices) < n:
        agregado = False
        for c in clases:
            if len(indices) >= n:
                break
            if ronda < len(por_clase[c]):
                indices.append(por_clase[c][ronda])
                agregado = True
        if not agregado:
            break
        ronda += 1
    return indices
