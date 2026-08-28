"""Prueba de humo del lote P1: modelos (VAE, GAN) y evaluar.

Verifica en CPU:
1. VAE de imagen entrena 10 épocas en < 3 min (requisito §D.2.2).
2. El VAE muestrea, codifica, decodifica e interpola.
3. El CVAE tabular genera muestras condicionadas de la clase minoritaria.
4. La GAN entrena y muestrea; con lr_d = 10*lr_g la cobertura de modos
   CAE respecto de la GAN equilibrada (la palanca de S2-E1 funciona).
5. tstr, fid_ligero y tabla_resultados devuelven lo pactado.

Guarda figuras en /tmp/humo_p1. Uso: python scripts/prueba_humo_p1.py
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import datos, modelos, evaluar, graficas  # noqa: E402

SALIDA = Path("/tmp/humo_p1")
SALIDA.mkdir(exist_ok=True)

torch.manual_seed(0)

# ══════════════════════════════════════════════════════════════
print("── 1 · VAE imagen: 10 épocas en < 180 s ──")
Xi, yi, mi = datos.cargar("imagen")
vae = modelos.VAE(mi, dim_latente=32)
inicio = time.time()
hist = vae.entrenar(Xi, epocas=10,
                    cb=lambda e, r: print(f"   época {e}: {r['total']:.1f}"))
t_vae = time.time() - inicio
print(f"  tiempo total: {t_vae:.0f} s")
assert t_vae < 180, f"VAE tardó {t_vae:.0f}s; el límite del plan es 180s"
assert hist["total"][-1] < hist["total"][0], "la pérdida no bajó"

muestras = vae.muestrear(16)
assert muestras.shape == (16, 1, 32, 32)
assert float(muestras.min()) >= -1 and float(muestras.max()) <= 1
graficas.rejilla(muestras, "VAE · 10 épocas").savefig(SALIDA / "vae_muestras.png")
graficas.curva(hist, "VAE").savefig(SALIDA / "vae_curva.png")

z = vae.codificar(Xi[:2])
assert z.shape == (2, 32)
graficas.interpolacion(vae, z[0], z[1], pasos=8).savefig(SALIDA / "vae_interp.png")
print("  ✓ muestrear / codificar / decodificar / interpolar")

# ══════════════════════════════════════════════════════════════
print("\n── 2 · CVAE tabular condicionado a la clase minoritaria ──")
Xt, yt, mt = datos.cargar("tabular")
cvae = modelos.VAE(mt, dim_latente=8, condicional=True)
cvae.entrenar(Xt, yt, epocas=15)
idx_min = mt["nombres_clases"].index(mt["clase_minoritaria"])
sint_min = cvae.muestrear(200, y=torch.full((200,), idx_min))
assert sint_min.shape == (200, 6)
graficas.comparar(Xt[yt == idx_min], sint_min,
                  ("falla real", "falla sintética")).savefig(
    SALIDA / "cvae_tabular.png")
print("  ✓ muestras condicionadas de la clase minoritaria")

# ══════════════════════════════════════════════════════════════
print("\n── 3 · GAN equilibrada vs GAN colapsada (la palanca de S2-E1) ──")
# RECETA CONGELADA DEL AULA (verificada con semillas 0, 1 y 2):
#   equilibrada: epocas=15 con lr por defecto (2e-4 ambos)
#   colapsada  : epocas=6 con lr_d = 10 * lr_g (1e-3 / 1e-4)
gan_ok = modelos.GAN(mi, dim_ruido=64)
inicio = time.time()
gan_ok.entrenar(Xi, epocas=15)
print(f"  GAN equilibrada: {time.time() - inicio:.0f} s")

gan_mala = modelos.GAN(mi, dim_ruido=64)
gan_mala.entrenar(Xi, epocas=6, lr_d=1e-3, lr_g=1e-4)  # lr_d = 10 * lr_g

s_ok = gan_ok.muestrear(500)
s_mala = gan_mala.muestrear(500)
cob_ok = evaluar.cobertura_modos(s_ok, Xi, k=5)
cob_mala = evaluar.cobertura_modos(s_mala, Xi, k=5)
print(f"  cobertura equilibrada: {cob_ok:.2f} · colapsada: {cob_mala:.2f}")
graficas.comparar(s_ok[:16], s_mala[:16],
                  ("GAN equilibrada", "GAN lr_d=10·lr_g")).savefig(
    SALIDA / "gan_colapso.png")
assert cob_mala < cob_ok, (
    "la GAN desbalanceada no perdió cobertura; revisar la palanca del colapso")
print("  ✓ el colapso reduce la cobertura de modos")

# ══════════════════════════════════════════════════════════════
print("\n── 4 · TSTR sobre tabular (clase foco = minoritaria) ──")
X_ent, y_ent, X_pru, y_pru = datos.dividir(Xt, yt)
y_sint = torch.full((400,), idx_min)
X_sint = cvae.muestrear(400, y=y_sint)
res = evaluar.tstr(X_ent, y_ent, X_sint, y_sint, X_pru, y_pru,
                   clase_foco=idx_min)
assert set(res) == {"linea_base", "tstr", "aumentado"}
evaluar.tabla_resultados(res)
print("  ✓ contrato de tstr")

# ══════════════════════════════════════════════════════════════
print("\n── 5 · fid_ligero ──")
d_igual = evaluar.fid_ligero(Xi[:400], Xi[400:800])
d_dist = evaluar.fid_ligero(Xi[:400], s_mala[:400])
print(f"  real vs real: {d_igual:.2f} · real vs colapsada: {d_dist:.2f}")
assert d_igual < d_dist, "fid_ligero no distingue real de colapsado"
print("  ✓ ordena correctamente")

print(f"\nTODAS LAS PRUEBAS DEL LOTE P1 PASARON (figuras en {SALIDA})")
