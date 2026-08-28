"""Celdas de rescate: nadie se queda atrás por un problema técnico.

Cada punto de fallo de los cuadernos tiene un checkpoint precomputado.
Si el entrenamiento no corrió (sin GPU, desconexión, error), se ejecuta:

    modelo, figuras = rescate.cargar("s1_vae")

y el ejercicio continúa desde ahí con resultados ya calculados.
"""

import io
from pathlib import Path

import torch

_RUTA_CHECKPOINTS = Path(__file__).resolve().parent.parent / "assets" / "checkpoints"

_URL_BASE = (
    "https://github.com/AdriannaGmz/IAA6_M13_Gen/"
    "releases/download/checkpoints/"
)

NOMBRES_VALIDOS = (
    "s1_vae",
    "s2_gan",
    "s2_gan_colapso",
    "s2_sintetico_generico",
    "s3_ddpm2d",
    # Añadido al resolver la decisión F.3 del plan: el «modelo de
    # difusión preentrenado» de S3-E2/S4-E1 es la clase Difusion del
    # módulo, preentrenada por la instructora sobre el conjunto demo.
    "s3_difusion_preentrenada",
    "s3_barrido",
    "s4_lora",
    "s4_tstr",
)


def hay_gpu():
    """Devuelve True/False e imprime el modo de ejecución activo."""
    if torch.cuda.is_available():
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        return True
    print("Modo CPU (degradado): todo funciona, sólo más lento. "
          "No es un error.")
    return False


def cargar(nombre):
    """Descarga y devuelve pesos y figuras precomputadas.

    nombres: s1_vae, s2_gan, s2_gan_colapso, s2_sintetico_generico,
             s3_ddpm2d, s3_barrido, s4_lora, s4_tstr

    Devuelve (contenido, figuras):
      contenido : lo guardado con ``guardar`` (dict con estado del
                  modelo, tensores o resultados, según el checkpoint).
      figuras   : lista de figuras de matplotlib reconstruidas de los
                  PNG precomputados (puede estar vacía).
    """
    if nombre not in NOMBRES_VALIDOS:
        raise ValueError(
            f"Checkpoint desconocido {nombre!r}. Válidos: {NOMBRES_VALIDOS}"
        )

    ruta_pt = _RUTA_CHECKPOINTS / f"{nombre}.pt"
    if not ruta_pt.exists():
        _descargar(f"{nombre}.pt", ruta_pt)

    contenido = torch.load(ruta_pt, map_location="cpu", weights_only=False)

    figuras = []
    for ruta_png in sorted(_RUTA_CHECKPOINTS.glob(f"{nombre}_fig*.png")):
        figuras.append(_png_a_figura(ruta_png))

    print(f"Checkpoint '{nombre}' cargado "
          f"({len(figuras)} figura(s) precomputada(s)). Puede continuar.")
    return contenido, figuras


def guardar(nombre, contenido=None, figuras=None, modelo=None):
    """Crea un checkpoint (uso de la instructora, no de los participantes).

    contenido : dict serializable con torch.save (tensores, listas,
                estados de modelo…).
    figuras   : lista de figuras de matplotlib; se guardan como PNG
                junto al .pt y ``cargar`` las reconstruye.
    modelo    : atajo — si se pasa un objeto con redes de torch, se
                guardan los state_dict de sus atributos nn.Module.
    """
    _RUTA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    contenido = dict(contenido or {})

    if modelo is not None:
        estados = {}
        for atributo, valor in vars(modelo).items():
            if isinstance(valor, torch.nn.Module):
                estados[atributo] = valor.state_dict()
        contenido.setdefault("state_dicts", estados)
        contenido.setdefault("clase_modelo", type(modelo).__name__)

    torch.save(contenido, _RUTA_CHECKPOINTS / f"{nombre}.pt")

    for i, fig in enumerate(figuras or []):
        fig.savefig(_RUTA_CHECKPOINTS / f"{nombre}_fig{i}.png",
                    dpi=110, bbox_inches="tight")
    print(f"Checkpoint '{nombre}' guardado en {_RUTA_CHECKPOINTS}")


# ──────────────────────────────────────────────────────────────

def _descargar(archivo, destino):
    import urllib.request

    url = _URL_BASE + archivo
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Descargando {archivo} …")
        urllib.request.urlretrieve(url, destino)
    except Exception as e:
        raise FileNotFoundError(
            f"No se encontró {archivo} localmente ni pudo descargarse "
            f"de {url}.\nSi está en clase: avise en el chat y siga con "
            "el cuaderno; este checkpoint no bloquea las celdas "
            "siguientes marcadas OBLIGATORIO."
        ) from e


def _png_a_figura(ruta_png):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    imagen = mpimg.imread(ruta_png)
    alto, ancho = imagen.shape[0], imagen.shape[1]
    fig, eje = plt.subplots(figsize=(ancho / 110, alto / 110))
    eje.imshow(imagen)
    eje.axis("off")
    fig.tight_layout(pad=0)
    return fig
