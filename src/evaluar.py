"""Métricas de evaluación de muestras sintéticas.

Contrato congelado (Plan de prácticas §A.2).

Referencias de los protocolos:
- TSTR: Esteban, Hyland y Rätsch (2017), «Real-valued (Medical) Time
  Series Generation with Recurrent Conditional GANs», arXiv:1706.02633.
  Los autores proponen y nombran el protocolo, y observan que la
  variante inversa no se degrada ante el colapso de modos mientras que
  TSTR sí.
- La distancia tipo Fréchet sigue la idea de Heusel et al.
  (NeurIPS 2017), pero con una red pequeña propia: NO es comparable con
  valores FID publicados; sólo sirve para comparaciones internas.
- ``cobertura_modos`` es una métrica DIDÁCTICA de este módulo
  (extensión nuestra): mide caída de cobertura por conteo de
  agrupamientos, en el espíritu de las métricas de precisión y
  cobertura de Kynkäänniemi et al. (arXiv:1904.06991) pero mucho más
  simple. No citarla como métrica publicada.
"""

import numpy as np
import torch


def _aplanar(X):
    X = X.detach().cpu() if isinstance(X, torch.Tensor) else torch.as_tensor(X)
    return X.reshape(len(X), -1).numpy().astype(np.float64)


def cobertura_modos(X_sint, X_real, k=5):
    """Fracción de modos reales representados en las muestras sintéticas.

    Es la métrica que se desploma en el ejercicio del colapso (S2-E1).

    Procedimiento (métrica didáctica, ver nota del módulo):
    1. Agrupa los datos reales en k modos con k-medias.
    2. Asigna cada muestra sintética a su modo más cercano.
    3. Un modo cuenta como cubierto si recibe al menos el 5 % de las
       muestras que le tocarían en un reparto uniforme.

    Devuelve un flotante en [0, 1]: 1.0 = todos los modos cubiertos.
    """
    from sklearn.cluster import KMeans

    A_real = _aplanar(X_real)
    A_sint = _aplanar(X_sint)

    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(A_real)
    asignacion = km.predict(A_sint)

    umbral = max(1, int(0.05 * len(A_sint) / k))
    conteo = np.bincount(asignacion, minlength=k)
    cubiertos = int((conteo >= umbral).sum())
    return cubiertos / k


def tstr(X_ent_real, y_ent_real, X_sint, y_sint, X_pru, y_pru,
         clase_foco=None):
    """Protocolo Train on Synthetic, Test on Real (Esteban et al., 2017).

    Entrena el mismo clasificador tres veces y evalúa siempre sobre los
    datos reales de prueba:
      - linea_base: entrenado sólo con reales.
      - tstr      : entrenado sólo con sintéticos.
      - aumentado : entrenado con reales + sintéticos.

    clase_foco: si se indica (índice entero de clase), el F1 reportado
    es el de esa clase — típicamente la minoritaria. Si no, F1 macro.

    Devuelve {"linea_base": f1, "tstr": f1, "aumentado": f1}.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    def _y(v):
        return (v.detach().cpu().numpy()
                if isinstance(v, torch.Tensor) else np.asarray(v))

    A_ent, ye = _aplanar(X_ent_real), _y(y_ent_real)
    A_sin, ys = _aplanar(X_sint), _y(y_sint)
    A_pru, yp = _aplanar(X_pru), _y(y_pru)

    conjuntos = {
        "linea_base": (A_ent, ye),
        "tstr": (A_sin, ys),
        "aumentado": (np.concatenate([A_ent, A_sin]),
                      np.concatenate([ye, ys])),
    }

    resultados = {}
    for nombre, (A, yy) in conjuntos.items():
        clf = RandomForestClassifier(n_estimators=100, random_state=0,
                                     n_jobs=-1)
        clf.fit(A, yy)
        pred = clf.predict(A_pru)
        if clase_foco is not None:
            f1 = f1_score(yp, pred, labels=[clase_foco],
                          average=None, zero_division=0)
            resultados[nombre] = round(float(f1[0]), 4)
        else:
            resultados[nombre] = round(
                float(f1_score(yp, pred, average="macro",
                               zero_division=0)), 4)
    return resultados


def fid_ligero(X_a, X_b):
    """Distancia tipo Fréchet sobre características de una red pequeña.

    NOTA: no comparable con valores publicados. Sólo comparación interna
    (mismo módulo, misma modalidad, misma corrida de aula).

    Para imágenes usa una red convolucional pequeña de pesos fijos
    (semilla determinista) como extractor; para tabular y series usa
    los datos aplanados. Devuelve un flotante >= 0.
    """
    from scipy import linalg

    Xa = X_a if isinstance(X_a, torch.Tensor) else torch.as_tensor(X_a)
    Xb = X_b if isinstance(X_b, torch.Tensor) else torch.as_tensor(X_b)
    Xa, Xb = Xa.float().cpu(), Xb.float().cpu()

    if Xa.ndim == 4:
        red = _extractor_fijo(Xa.shape[1])
        with torch.no_grad():
            Fa = red(Xa).numpy().astype(np.float64)
            Fb = red(Xb).numpy().astype(np.float64)
    else:
        Fa, Fb = _aplanar(Xa), _aplanar(Xb)

    mu_a, mu_b = Fa.mean(0), Fb.mean(0)
    cov_a = np.cov(Fa, rowvar=False) + 1e-6 * np.eye(Fa.shape[1])
    cov_b = np.cov(Fb, rowvar=False) + 1e-6 * np.eye(Fb.shape[1])

    raiz = linalg.sqrtm(cov_a @ cov_b)
    if np.iscomplexobj(raiz):
        raiz = raiz.real
    d2 = float(((mu_a - mu_b) ** 2).sum()
               + np.trace(cov_a + cov_b - 2 * raiz))
    return max(d2, 0.0)


_EXTRACTOR = {}


def _extractor_fijo(canales):
    """Red convolucional pequeña de pesos fijos (no entrenada)."""
    import torch.nn as nn

    if canales not in _EXTRACTOR:
        torch.manual_seed(1234)
        _EXTRACTOR[canales] = nn.Sequential(
            nn.Conv2d(canales, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),                      # 32 * 2 * 2 = 128 rasgos
        ).eval()
    return _EXTRACTOR[canales]


def tabla_resultados(dic):
    """Formatea el diccionario como tabla lista para pegar en la bitácora.

    Acepta un dict plano {nombre: valor} o anidado
    {renglon: {columna: valor}}. Imprime y devuelve la tabla en Markdown.
    """
    def _celda(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    anidado = all(isinstance(v, dict) for v in dic.values()) and dic
    if anidado:
        columnas = list(next(iter(dic.values())).keys())
        lineas = ["| | " + " | ".join(columnas) + " |",
                  "|---" * (len(columnas) + 1) + "|"]
        for renglon, valores in dic.items():
            lineas.append(
                "| **" + str(renglon) + "** | "
                + " | ".join(_celda(valores.get(c, "")) for c in columnas)
                + " |")
    else:
        lineas = ["| protocolo | F1 |", "|---|---|"]
        for nombre, valor in dic.items():
            lineas.append(f"| {nombre} | {_celda(valor)} |")

    tabla = "\n".join(lineas)
    print(tabla)
    return tabla
