"""Prueba de humo del lote P3: Difusion y adaptación LoRA.

Verifica en CPU:
1. Difusion tabular condicional: entrena 5 épocas, la pérdida baja,
   muestrea con guía y devuelve trayectoria.
2. Difusion de imagen: entrena 2 épocas y muestrea a 5 y 50 pasos SIN
   divergir (regresión del bug del ruido inconsistente con el x0
   recortado: antes |x| → NaN con muchos pasos).
3. El muestreo es determinista dado el mismo ruido inicial.
4. peft envuelve la red (LoRA r=8) con < 10 % de parámetros
   entrenables, entrena un paso y muestrea.
5. Velocidad: muestrear 8 imágenes a 50 pasos < 20 s por muestra
   (requisito de la decisión F.3).

Guarda figuras en /tmp/humo_p3. Uso: python scripts/prueba_humo_p3.py
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import datos, modelos, graficas  # noqa: E402

SALIDA = Path("/tmp/humo_p3")
SALIDA.mkdir(exist_ok=True)

torch.manual_seed(0)

# ══════════════════════════════════════════════════════════════
print("── 1 · Difusion tabular condicional ──")
Xt, yt, mt = datos.cargar("tabular")
d = modelos.Difusion(mt, pasos=100)
hist = d.entrenar(Xt, yt, epocas=5)
assert hist["perdida"][-1] < hist["perdida"][0], "la pérdida no bajó"
idx_min = mt["nombres_clases"].index(mt["clase_minoritaria"])
s = d.muestrear(50, y=idx_min, pasos=25, guia=3.0)
assert s.shape == (50, Xt.shape[1]), s.shape
assert torch.isfinite(s).all(), "muestras no finitas"
m, tray = d.muestrear(10, pasos=10, guardar_trayectoria=True)
assert len(tray) == 11, "trayectoria incompleta"
print("  OK: condicional, guía y trayectoria")

# ══════════════════════════════════════════════════════════════
print("── 2 · Difusion imagen: estable a 5 y a 50 pasos ──")
Xi, yi, mi = datos.cargar("imagen")
di = modelos.Difusion(mi, pasos=200)
di.entrenar(Xi[:512], yi[:512], epocas=2)
for pasos in (5, 50):
    torch.manual_seed(3)
    im = di.muestrear(8, y=0, pasos=pasos, guia=3.0)
    assert im.shape == (8, 1, 32, 32), im.shape
    assert torch.isfinite(im).all(), f"divergió a {pasos} pasos"
    assert im.abs().max() <= 1.0 + 1e-5, "fuera de rango [-1, 1]"
print("  OK: sin divergencia (regresión del ruido inconsistente)")

# determinismo del muestreo
torch.manual_seed(3)
a = di.muestrear(4, y=0, pasos=25, guia=3.0)
torch.manual_seed(3)
b = di.muestrear(4, y=0, pasos=25, guia=3.0)
assert torch.allclose(a, b), "el muestreo no es determinista"
print("  OK: muestreo determinista con la misma semilla")

# ══════════════════════════════════════════════════════════════
print("── 3 · LoRA (peft) sobre la red de Difusion ──")
from peft import LoraConfig, get_peft_model  # noqa: E402

config = LoraConfig(r=8, lora_alpha=16,
                    target_modules=["baja1.0", "baja2.0", "baja3.0",
                                    "baja4.0", "medio.0", "medio.2",
                                    "salida", "p2", "p3", "p4"],
                    modules_to_save=["emb"])
di.red = get_peft_model(di.red, config)
entrenables = sum(p.numel() for p in di.red.parameters()
                  if p.requires_grad)
total = sum(p.numel() for p in di.red.parameters())
frac = entrenables / total
print(f"  entrenables: {entrenables} de {total} ({frac:.1%})")
assert frac < 0.10, "el adaptador entrena demasiados parámetros"
di.entrenar(Xi[:256], yi[:256], epocas=1)
im = di.muestrear(4, y=0, pasos=10)
assert torch.isfinite(im).all()
print("  OK: adaptar y muestrear con LoRA activo")

# ══════════════════════════════════════════════════════════════
print("── 4 · Velocidad de muestreo (decisión F.3: < 20 s/muestra) ──")
inicio = time.time()
im = di.muestrear(8, y=0, pasos=50, guia=3.0)
por_muestra = (time.time() - inicio) / 8
print(f"  {por_muestra:.2f} s por muestra a 50 pasos")
assert por_muestra < 20, "demasiado lento para el aula"
fig = graficas.rejilla(im, "humo P3: 8 muestras", n=8)
fig.savefig(SALIDA / "humo_p3_muestras.png", dpi=110)

print("\nPRUEBA DE HUMO P3: TODO EN ORDEN")
