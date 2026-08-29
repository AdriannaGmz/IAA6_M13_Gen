"""Prueba de humo del lote P0: datos.py y graficas.py completos.

Ejercita cada función pública con las tres modalidades y guarda las
figuras en /tmp/humo_p0 para inspección visual.

Uso:  python scripts/prueba_humo_p0.py
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import datos, graficas, rescate  # noqa: E402

SALIDA = Path("/tmp/humo_p0")
SALIDA.mkdir(exist_ok=True)


def prueba(nombre, funcion):
    inicio = time.time()
    resultado = funcion()
    print(f"  ✓ {nombre} ({time.time() - inicio:.1f}s)")
    return resultado


print("── rescate.hay_gpu ──")
rescate.hay_gpu()

for modo in ("imagen", "tabular", "serie"):
    print(f"\n── modalidad: {modo} ──")
    inicio = time.time()
    X, y, meta = datos.cargar(modo)
    t_carga = time.time() - inicio
    assert t_carga < 10, f"carga de {modo} tardó {t_carga:.1f}s (límite 10s)"
    print(f"  ✓ cargar en {t_carga:.2f}s · forma {meta['forma']}")

    assert float(X.min()) >= -1.0001 and float(X.max()) <= 1.0001, "fuera de [-1,1]"
    assert meta["desbalance"] < 0.10, "la clase minoritaria no es minoritaria"
    for clave in ("n_muestras", "n_clases", "nombres_clases", "conteo_por_clase",
                  "clase_minoritaria", "desbalance", "modalidad", "forma", "fuente"):
        assert clave in meta, f"falta clave {clave} en meta"
    print(f"  ✓ contrato de meta completo · minoritaria: "
          f"{meta['clase_minoritaria']} ({meta['desbalance']:.1%})")

    X_ent, y_ent, X_pru, y_pru = datos.dividir(X, y)
    frac_ent = float((y_ent == y_ent.unique()[torch.argmin(torch.bincount(y_ent))]).float().mean())
    assert len(X_ent) + len(X_pru) == len(X)
    # estratificación: la proporción minoritaria se conserva ±1 punto
    prop_total = meta["desbalance"]
    prop_pru = float(torch.bincount(y_pru).min()) / len(y_pru)
    assert abs(prop_total - prop_pru) < 0.01, "la partición no está estratificada"
    print(f"  ✓ dividir estratificado ({len(X_ent)} ent / {len(X_pru)} pru)")

    texto = datos.ficha(meta)
    assert meta["clase_minoritaria"] in texto

    fig = datos.vistazo(X, y, meta)
    fig.savefig(SALIDA / f"vistazo_{modo}.png", dpi=100)
    print(f"  ✓ vistazo guardado en {SALIDA}/vistazo_{modo}.png")

    # desnormalización regresa a unidades razonables
    X_orig = datos.desnormalizar(X[:5], meta)
    assert torch.isfinite(X_orig).all()

print("\n── graficas ──")
Xi, yi, mi = datos.cargar("imagen")
Xt, yt, mt = datos.cargar("tabular")
Xs, ys, ms = datos.cargar("serie")

prueba("rejilla", lambda: graficas.rejilla(Xi[:16], "demostración").savefig(SALIDA / "rejilla.png"))
prueba("curva (dict)", lambda: graficas.curva(
    {"perdida_d": [0.9, 0.7, 0.6, 0.55], "perdida_g": [1.2, 1.0, 1.1, 1.3]},
    "GAN").savefig(SALIDA / "curva.png"))
prueba("comparar imagen", lambda: graficas.comparar(Xi[:16], Xi[16:32], ("VAE", "GAN")).savefig(SALIDA / "comparar_imagen.png"))
prueba("comparar tabular", lambda: graficas.comparar(Xt[:500], Xt[500:1000]).savefig(SALIDA / "comparar_tabular.png"))
prueba("comparar serie", lambda: graficas.comparar(Xs[:8], Xs[8:16]).savefig(SALIDA / "comparar_serie.png"))

resultados = {(p, g): Xi[i] for i, (p, g) in enumerate(
    [(p, g) for g in (1, 3, 7.5) for p in (5, 10, 25)])}
prueba("rejilla_barrido", lambda: graficas.rejilla_barrido(
    resultados, "pasos", "guia").savefig(SALIDA / "barrido.png"))

nubes = [torch.randn(400, 2) * (1 - a) + a * torch.stack(
    [torch.cos(torch.linspace(0, 6.28, 400)),
     torch.sin(torch.linspace(0, 6.28, 400))], dim=1)
    for a in (0.0, 0.3, 0.6, 0.9, 1.0)]
prueba("trayectoria_2d", lambda: graficas.trayectoria_2d(nubes).savefig(SALIDA / "trayectoria.png"))


class _DecodificadorFalso:
    """Simula modelo.decodificar para probar interpolacion sin VAE."""

    def decodificar(self, z):
        n = len(z)
        return Xi[:n]


prueba("interpolacion (imagen)", lambda: graficas.interpolacion(
    _DecodificadorFalso(), torch.randn(32), torch.randn(32), pasos=8
).savefig(SALIDA / "interpolacion.png"))

# rescate: guardar y cargar un checkpoint de ida y vuelta.
# Se redirige el directorio a /tmp: con el de producción, esta prueba
# sobrescribe el checkpoint real de s1_vae que S2-E1 necesita.
print("\n── rescate (ida y vuelta) ──")
rescate._RUTA_CHECKPOINTS = SALIDA / "checkpoints"
figs = [graficas.rejilla(Xi[:4], "prueba", n=4)]
rescate.guardar("s1_vae", contenido={"prueba": torch.arange(3)}, figuras=figs)
contenido, figuras = rescate.cargar("s1_vae")
assert torch.equal(contenido["prueba"], torch.arange(3))
assert len(figuras) == 1
print("  ✓ guardar/cargar checkpoint")

print("\nTODAS LAS PRUEBAS DEL LOTE P0 PASARON")
