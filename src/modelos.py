"""Las tres familias generativas con la MISMA firma.

Contrato congelado (Plan de prácticas §A.2). La simetría es deliberada:
el participante cambia ``VAE`` por ``GAN`` o por ``Difusion`` y el resto
de su cuaderno no se toca.

Convención común
----------------
- ``meta`` es el diccionario que devuelve ``datos.cargar``: de ahí salen
  la modalidad y la forma de los datos; el participante no describe su
  arquitectura a mano.
- ``entrenar`` recibe X en [-1, 1] y devuelve un historial
  (dict de listas) que ``graficas.curva`` grafica directamente.
- ``muestrear`` devuelve tensores en [-1, 1] con la misma forma que X.
- ``cb`` es un callback opcional ``cb(epoca, registro)`` para barras de
  progreso.
- Sin GPU todo funciona igual, sólo más lento (modo degradado).

Referencias de las arquitecturas:
- VAE: Kingma y Welling, «Auto-Encoding Variational Bayes», ICLR 2014.
- GAN: Goodfellow et al., «Generative Adversarial Nets», NeurIPS 2014;
  convenciones convolucionales de Radford, Metz y Chintala (DCGAN,
  ICLR 2016) adaptadas a 32×32.
- Difusion: Ho, Jain y Abbeel, «Denoising Diffusion Probabilistic
  Models», NeurIPS 2020; muestreo con pocos pasos según Song, Meng y
  Ermon (DDIM, ICLR 2021); guía sin clasificador según Ho y Salimans
  (taller de NeurIPS 2021).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _dispositivo():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _forma_muestra(meta):
    """Forma de UNA muestra (sin la dimensión N)."""
    return tuple(meta["forma"][1:])


def _un_caliente(y, n_clases):
    return F.one_hot(y.long(), n_clases).float()


def _lotes(X, y, tam_lote, generador):
    idx = torch.randperm(len(X), generator=generador)
    for i in range(0, len(X), tam_lote):
        sel = idx[i:i + tam_lote]
        yield X[sel], (y[sel] if y is not None else None)


# ══════════════════════════════════════════════════════════════
# VAE
# ══════════════════════════════════════════════════════════════

class _CodificadorConv(nn.Module):
    def __init__(self, canales, dim_extra, dim_latente):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(canales, 32, 4, 2, 1), nn.ReLU(),    # 32 -> 16
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),         # 16 -> 8
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),        # 8  -> 4
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(128 * 4 * 4 + dim_extra, dim_latente)
        self.fc_logvar = nn.Linear(128 * 4 * 4 + dim_extra, dim_latente)

    def forward(self, x, cond=None):
        h = self.convs(x)
        if cond is not None:
            h = torch.cat([h, cond], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class _DecodificadorConv(nn.Module):
    def __init__(self, canales, dim_entrada):
        super().__init__()
        self.fc = nn.Linear(dim_entrada, 128 * 4 * 4)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),   # 4  -> 8
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),    # 8  -> 16
            nn.ConvTranspose2d(32, canales, 4, 2, 1), nn.Tanh(),  # 16 -> 32
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 128, 4, 4)
        return self.deconvs(h)


class _CodificadorMLP(nn.Module):
    def __init__(self, dim_datos, dim_extra, dim_latente):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(dim_datos + dim_extra, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, dim_latente)
        self.fc_logvar = nn.Linear(128, dim_latente)

    def forward(self, x, cond=None):
        h = x.flatten(1)
        if cond is not None:
            h = torch.cat([h, cond], dim=1)
        h = self.red(h)
        return self.fc_mu(h), self.fc_logvar(h)


class _DecodificadorMLP(nn.Module):
    def __init__(self, dim_datos, dim_entrada, forma_salida):
        super().__init__()
        self.forma_salida = forma_salida
        self.red = nn.Sequential(
            nn.Linear(dim_entrada, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, dim_datos), nn.Tanh(),
        )

    def forward(self, z):
        salida = self.red(z)
        return salida.view(-1, *self.forma_salida)


class VAE:
    """Autocodificador variacional (variational autoencoder).

    Kingma y Welling (ICLR 2014). Se construye en S1-E2 y reaparece como
    línea de comparación en S2-E1. Con ``condicional=True`` es un CVAE:
    la etiqueta de clase entra al codificador y al decodificador
    (S2-E2).
    """

    def __init__(self, meta, dim_latente=32, condicional=False, semilla=0):
        # semilla fija: todo el grupo ve el mismo comportamiento en clase
        torch.manual_seed(semilla)
        self.meta = meta
        self.dim_latente = dim_latente
        self.condicional = condicional
        self.n_clases = meta["n_clases"]
        self.dispositivo = _dispositivo()
        self._gen = torch.Generator().manual_seed(semilla)

        forma = _forma_muestra(meta)
        dim_extra = self.n_clases if condicional else 0

        if meta["modalidad"] == "imagen":
            canales = forma[0]
            self.codificador = _CodificadorConv(canales, dim_extra, dim_latente)
            self.decodificador = _DecodificadorConv(
                canales, dim_latente + dim_extra)
        else:  # tabular y serie comparten el esqueleto MLP
            dim_datos = int(np.prod(forma))
            self.codificador = _CodificadorMLP(dim_datos, dim_extra, dim_latente)
            self.decodificador = _DecodificadorMLP(
                dim_datos, dim_latente + dim_extra, forma)

        self.codificador.to(self.dispositivo)
        self.decodificador.to(self.dispositivo)

    # ── entrenamiento ────────────────────────────────────────

    def entrenar(self, X, y=None, epocas=10, cb=None):
        """Optimiza el objetivo variacional: reconstrucción + regularización.

        Devuelve historial {"total", "reconstruccion", "regularizacion"},
        con el promedio por muestra de cada época.
        """
        if self.condicional and y is None:
            raise ValueError("El modelo es condicional: pase y al entrenar.")
        X = X.to(self.dispositivo)
        y = y.to(self.dispositivo) if y is not None else None

        opt = torch.optim.Adam(
            list(self.codificador.parameters())
            + list(self.decodificador.parameters()), lr=1e-3)

        historial = {"total": [], "reconstruccion": [], "regularizacion": []}
        for epoca in range(epocas):
            suma_rec, suma_reg, n_vistos = 0.0, 0.0, 0
            for x_lote, y_lote in _lotes(X, y, 128, self._gen):
                cond = (_un_caliente(y_lote, self.n_clases)
                        if self.condicional else None)
                mu, logvar = self.codificador(x_lote, cond)
                # truco de reparametrización: z = mu + sigma * epsilon
                epsilon = torch.randn_like(mu)
                z = mu + torch.exp(0.5 * logvar) * epsilon
                entrada_dec = (torch.cat([z, cond], dim=1)
                               if self.condicional else z)
                x_rec = self.decodificador(entrada_dec)

                rec = F.mse_loss(x_rec, x_lote, reduction="sum") / len(x_lote)
                reg = (-0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp()) / len(x_lote))
                perdida = rec + reg

                opt.zero_grad()
                perdida.backward()
                opt.step()

                suma_rec += float(rec.detach()) * len(x_lote)
                suma_reg += float(reg.detach()) * len(x_lote)
                n_vistos += len(x_lote)

            historial["reconstruccion"].append(suma_rec / n_vistos)
            historial["regularizacion"].append(suma_reg / n_vistos)
            historial["total"].append((suma_rec + suma_reg) / n_vistos)
            if cb:
                cb(epoca, {k: v[-1] for k, v in historial.items()})
        return historial

    # ── uso ──────────────────────────────────────────────────

    def muestrear(self, n, y=None):
        """Genera n muestras decodificando latentes de la previa N(0, I)."""
        z = torch.randn(n, self.dim_latente, device=self.dispositivo)
        if self.condicional:
            if y is None:
                y = torch.randint(0, self.n_clases, (n,))
            y = torch.as_tensor(y, device=self.dispositivo)
            if y.ndim == 0:
                y = y.repeat(n)
            return self.decodificar(z, y)
        return self.decodificar(z)

    def codificar(self, X):
        """Devuelve la media del latente para cada muestra de X."""
        with torch.no_grad():
            X = X.to(self.dispositivo)
            cond = None
            if self.condicional:
                raise ValueError(
                    "Para un CVAE use codificar_condicional(X, y).")
            mu, _ = self.codificador(X, cond)
        return mu.cpu()

    def codificar_condicional(self, X, y):
        """Media del latente para un CVAE (X con sus etiquetas y)."""
        with torch.no_grad():
            X = X.to(self.dispositivo)
            cond = _un_caliente(
                torch.as_tensor(y, device=self.dispositivo), self.n_clases)
            mu, _ = self.codificador(X, cond)
        return mu.cpu()

    def decodificar(self, z, y=None):
        """Decodifica un lote de latentes z -> espacio de datos [-1, 1].

        Para un modelo condicional debe pasarse y (etiquetas enteras).
        """
        with torch.no_grad():
            z = torch.as_tensor(z, dtype=torch.float32,
                                device=self.dispositivo)
            if z.ndim == 1:
                z = z[None, :]
            if self.condicional:
                if y is None:
                    raise ValueError(
                        "El modelo es condicional: pase y a decodificar.")
                y = torch.as_tensor(y, device=self.dispositivo)
                if y.ndim == 0:
                    y = y.repeat(len(z))
                cond = _un_caliente(y, self.n_clases)
                z = torch.cat([z, cond], dim=1)
            return self.decodificador(z).cpu()


# ══════════════════════════════════════════════════════════════
# GAN
# ══════════════════════════════════════════════════════════════

class _GeneradorConv(nn.Module):
    def __init__(self, canales, dim_entrada):
        super().__init__()
        self.fc = nn.Linear(dim_entrada, 128 * 4 * 4)
        self.red = nn.Sequential(
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),                  # 4 -> 8
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),                   # 8 -> 16
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, canales, 4, 2, 1), nn.Tanh(),   # 16 -> 32
        )

    def forward(self, z):
        return self.red(self.fc(z).view(-1, 128, 4, 4))


class _DiscriminadorConv(nn.Module):
    def __init__(self, canales, n_clases_cond):
        super().__init__()
        self.n_clases_cond = n_clases_cond
        entrada = canales + n_clases_cond   # condición como mapas constantes
        self.red = nn.Sequential(
            nn.Conv2d(entrada, 32, 4, 2, 1), nn.LeakyReLU(0.2),    # 32 -> 16
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),         # 16 -> 8
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),        # 8  -> 4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1),
        )

    def forward(self, x, cond=None):
        if cond is not None:
            mapas = cond[:, :, None, None].expand(
                -1, -1, x.shape[2], x.shape[3])
            x = torch.cat([x, mapas], dim=1)
        return self.red(x).squeeze(1)


class _GeneradorMLP(nn.Module):
    def __init__(self, dim_datos, dim_entrada, forma_salida):
        super().__init__()
        self.forma_salida = forma_salida
        self.red = nn.Sequential(
            nn.Linear(dim_entrada, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, dim_datos), nn.Tanh(),
        )

    def forward(self, z):
        return self.red(z).view(-1, *self.forma_salida)


class _DiscriminadorMLP(nn.Module):
    def __init__(self, dim_datos, n_clases_cond):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(dim_datos + n_clases_cond, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x, cond=None):
        h = x.flatten(1)
        if cond is not None:
            h = torch.cat([h, cond], dim=1)
        return self.red(h).squeeze(1)


class GAN:
    """Red generativa antagónica (generative adversarial network).

    Goodfellow et al. (NeurIPS 2014); para imágenes, convenciones de
    DCGAN (Radford, Metz y Chintala, ICLR 2016) adaptadas a 32×32.

    lr_d y lr_g están expuestos DELIBERADAMENTE: son la palanca con la
    que S2-E1 provoca el colapso de modos (lr_d = 10 * lr_g).
    """

    def __init__(self, meta, dim_ruido=64, condicional=False, semilla=0):
        # semilla fija: todo el grupo ve el mismo comportamiento en clase
        torch.manual_seed(semilla)
        self.meta = meta
        self.dim_ruido = dim_ruido
        self.condicional = condicional
        self.n_clases = meta["n_clases"]
        self.dispositivo = _dispositivo()
        self._gen = torch.Generator().manual_seed(semilla)

        forma = _forma_muestra(meta)
        dim_extra = self.n_clases if condicional else 0
        n_cond_d = self.n_clases if condicional else 0

        if meta["modalidad"] == "imagen":
            canales = forma[0]
            self.generador = _GeneradorConv(canales, dim_ruido + dim_extra)
            self.discriminador = _DiscriminadorConv(canales, n_cond_d)
        else:
            dim_datos = int(np.prod(forma))
            self.generador = _GeneradorMLP(
                dim_datos, dim_ruido + dim_extra, forma)
            self.discriminador = _DiscriminadorMLP(dim_datos, n_cond_d)

        self.generador.to(self.dispositivo)
        self.discriminador.to(self.dispositivo)

    def entrenar(self, X, y=None, epocas=20, lr_d=2e-4, lr_g=2e-4, cb=None):
        """Juego minimax por lotes (pérdida BCE no saturante para G).

        Estabilización: las etiquetas reales del discriminador se
        suavizan a 0.9 (one-sided label smoothing, Salimans et al.,
        «Improved techniques for training GANs», NeurIPS 2016).

        Devuelve historial {"perdida_d", "perdida_g"} por época.
        """
        if self.condicional and y is None:
            raise ValueError("El modelo es condicional: pase y al entrenar.")
        X = X.to(self.dispositivo)
        y = y.to(self.dispositivo) if y is not None else None

        opt_d = torch.optim.Adam(self.discriminador.parameters(),
                                 lr=lr_d, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generador.parameters(),
                                 lr=lr_g, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()

        historial = {"perdida_d": [], "perdida_g": []}
        for epoca in range(epocas):
            suma_d, suma_g, n_lotes = 0.0, 0.0, 0
            for x_real, y_lote in _lotes(X, y, 128, self._gen):
                n = len(x_real)
                cond = (_un_caliente(y_lote, self.n_clases)
                        if self.condicional else None)

                # ── discriminador ──
                z = torch.randn(n, self.dim_ruido, device=self.dispositivo)
                entrada_g = (torch.cat([z, cond], dim=1)
                             if self.condicional else z)
                x_falso = self.generador(entrada_g).detach()

                logits_real = self.discriminador(x_real, cond)
                logits_falso = self.discriminador(x_falso, cond)
                perdida_d = (bce(logits_real,
                                 0.9 * torch.ones_like(logits_real))
                             + bce(logits_falso,
                                   torch.zeros_like(logits_falso)))
                opt_d.zero_grad()
                perdida_d.backward()
                opt_d.step()

                # ── generador (no saturante) ──
                z = torch.randn(n, self.dim_ruido, device=self.dispositivo)
                entrada_g = (torch.cat([z, cond], dim=1)
                             if self.condicional else z)
                x_falso = self.generador(entrada_g)
                logits = self.discriminador(x_falso, cond)
                perdida_g = bce(logits, torch.ones_like(logits))
                opt_g.zero_grad()
                perdida_g.backward()
                opt_g.step()

                suma_d += float(perdida_d.detach())
                suma_g += float(perdida_g.detach())
                n_lotes += 1

            historial["perdida_d"].append(suma_d / n_lotes)
            historial["perdida_g"].append(suma_g / n_lotes)
            if cb:
                cb(epoca, {k: v[-1] for k, v in historial.items()})
        return historial

    def muestrear(self, n, y=None):
        """Genera n muestras a partir de ruido; y opcional si es condicional."""
        with torch.no_grad():
            z = torch.randn(n, self.dim_ruido, device=self.dispositivo)
            if self.condicional:
                if y is None:
                    y = torch.randint(0, self.n_clases, (n,))
                y = torch.as_tensor(y, device=self.dispositivo)
                if y.ndim == 0:
                    y = y.repeat(n)
                cond = _un_caliente(y, self.n_clases)
                z = torch.cat([z, cond], dim=1)
            return self.generador(z).cpu()


# ══════════════════════════════════════════════════════════════
# Difusión
# ══════════════════════════════════════════════════════════════

class _EmbebidoTiempoClase(nn.Module):
    """Embebido conjunto del paso de tiempo t y la clase (con clase nula).

    La clase n_clases (la última) es la clase NULA: se usa durante el
    entrenamiento (abandono de condición) y para la rama incondicional
    de la guía sin clasificador (Ho y Salimans, 2021/2022).
    """

    def __init__(self, dim, n_clases):
        super().__init__()
        self.dim = dim
        self.clases = nn.Embedding(n_clases + 1, dim)
        self.red = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(),
                                 nn.Linear(dim, dim))

    def forward(self, t, y):
        # embebido sinusoidal del tiempo (Ho, Jain y Abbeel, 2020)
        mitad = self.dim // 2
        frec = torch.exp(-np.log(10000.0)
                         * torch.arange(mitad, device=t.device) / mitad)
        ang = t.float()[:, None] * frec[None, :]
        emb_t = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        return self.red(torch.cat([emb_t, self.clases(y)], dim=1))


class _RuidoConv(nn.Module):
    """Red quita-ruido convolucional mínima con saltos (imágenes 32x32).

    Baja hasta 4x4 para que el cuello tenga contexto GLOBAL de la
    imagen: sin él, el modelo aprende la textura local pero no la
    disposición de la figura (verificado en producción: con fondo de
    8x8 las muestras salían amorfas).
    """

    def __init__(self, canales, n_clases, dim_emb=64):
        super().__init__()
        self.emb = _EmbebidoTiempoClase(dim_emb, n_clases)
        self.baja1 = nn.Sequential(nn.Conv2d(canales, 32, 3, 1, 1),
                                   nn.ReLU())                    # 32
        self.baja2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1),
                                   nn.ReLU())                    # 16
        self.baja3 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1),
                                   nn.ReLU())                    # 8
        self.baja4 = nn.Sequential(nn.Conv2d(128, 128, 4, 2, 1),
                                   nn.ReLU())                    # 4
        self.medio = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 128, 3, 1, 1),
                                   nn.ReLU())
        self.sube3 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1),
                                   nn.ReLU())                    # 8
        self.sube2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1),
                                   nn.ReLU())                    # 16
        self.sube1 = nn.Sequential(nn.ConvTranspose2d(128, 32, 4, 2, 1),
                                   nn.ReLU())                    # 32
        self.salida = nn.Conv2d(64, canales, 3, 1, 1)
        self.p2 = nn.Linear(dim_emb, 64)
        self.p3 = nn.Linear(dim_emb, 128)
        self.p4 = nn.Linear(dim_emb, 128)

    def forward(self, x, t, y):
        e = self.emb(t, y)
        h1 = self.baja1(x)                                       # 32, 32c
        h2 = self.baja2(h1) + self.p2(e)[:, :, None, None]       # 16, 64c
        h3 = self.baja3(h2) + self.p3(e)[:, :, None, None]       # 8, 128c
        h4 = self.baja4(h3) + self.p4(e)[:, :, None, None]       # 4, 128c
        m = self.medio(h4)
        s3 = self.sube3(m)                                       # 8, 128c
        s2 = self.sube2(torch.cat([s3, h3], dim=1))              # 16, 64c
        s1 = self.sube1(torch.cat([s2, h2], dim=1))              # 32, 32c
        return self.salida(torch.cat([s1, h1], dim=1))


class _RuidoMLP(nn.Module):
    """Red quita-ruido de perceptrón (tabular, series y juguetes 2D)."""

    def __init__(self, dim_datos, n_clases, dim_emb=64):
        super().__init__()
        self.emb = _EmbebidoTiempoClase(dim_emb, n_clases)
        self.red = nn.Sequential(
            nn.Linear(dim_datos + dim_emb, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, dim_datos),
        )

    def forward(self, x, t, y):
        h = torch.cat([x.flatten(1), self.emb(t, y)], dim=1)
        return self.red(h).view(x.shape)


class Difusion:
    """Modelo de difusión quita-ruido (denoising diffusion model).

    Ho, Jain y Abbeel, «Denoising Diffusion Probabilistic Models»,
    NeurIPS 2020. El objetivo de entrenamiento es el SIMPLIFICADO del
    artículo (error cuadrático medio sobre el ruido predicho), que los
    autores adoptan tras descartar los términos de ponderación del
    objetivo variacional y justifican por desempeño empírico.

    - La parte DISEÑADA: el calendario de ruido (betas lineales
      1e-4 → 0.02, como en el artículo) y la distribución final.
    - La parte APRENDIDA: la red que predice el ruido.
    - Muestreo con menos pasos: regla determinista de DDIM (Song, Meng
      y Ermon, ICLR 2021) sobre un subcalendario.
    - Guía sin clasificador (Ho y Salimans, taller NeurIPS 2021):
      durante el entrenamiento la condición se abandona el 10 % de las
      veces; al muestrear,
      ruido = ruido_nulo + guia * (ruido_condicionado - ruido_nulo).
    """

    def __init__(self, meta, pasos=200, semilla=0):
        torch.manual_seed(semilla)
        self.meta = meta
        self.pasos = pasos
        self.n_clases = meta["n_clases"]
        self.dispositivo = _dispositivo()
        self._gen = torch.Generator().manual_seed(semilla)

        forma = _forma_muestra(meta)
        if meta["modalidad"] == "imagen":
            self.red = _RuidoConv(forma[0], self.n_clases)
        else:
            self.red = _RuidoMLP(int(np.prod(forma)), self.n_clases)
        self.red.to(self.dispositivo)

        # calendario fijo (la parte diseñada)
        betas = torch.linspace(1e-4, 0.02, pasos)
        self.alfas_cum = torch.cumprod(1.0 - betas, dim=0).to(
            self.dispositivo)

    def _y_o_nula(self, y, n):
        """Etiquetas al dispositivo; None → clase nula (incondicional)."""
        if y is None:
            return torch.full((n,), self.n_clases,
                              device=self.dispositivo, dtype=torch.long)
        y = torch.as_tensor(y, device=self.dispositivo, dtype=torch.long)
        if y.ndim == 0:
            y = y.repeat(n)
        return y

    def entrenar(self, X, y=None, epocas=30, cb=None, tam_lote=128,
                 lr=1e-3):
        """Minimiza el objetivo simplificado de DDPM.

        Si recibe ``y``, entrena condicional con abandono del 10 %
        (necesario para la guía sin clasificador). Devuelve historial
        {"perdida": [...]}. lr=1e-3 va bien para esta red pequeña.
        """
        opt = torch.optim.Adam(self.red.parameters(), lr=lr)
        historial = {"perdida": []}
        for epoca in range(epocas):
            suma, n_vistos = 0.0, 0
            for x_lote, y_lote in _lotes(X, y, tam_lote, self._gen):
                x0 = x_lote.to(self.dispositivo)
                nb = len(x0)
                yb = self._y_o_nula(y_lote, nb)
                # abandono de condición: 10 % de los casos van a la nula
                mascara = torch.rand(nb, device=self.dispositivo) < 0.1
                yb = torch.where(mascara,
                                 torch.full_like(yb, self.n_clases), yb)

                t = torch.randint(0, self.pasos, (nb,),
                                  device=self.dispositivo)
                ac = self.alfas_cum[t].view(-1, *([1] * (x0.ndim - 1)))
                ruido = torch.randn_like(x0)
                x_t = ac.sqrt() * x0 + (1 - ac).sqrt() * ruido

                perdida = F.mse_loss(self.red(x_t, t, yb), ruido)
                opt.zero_grad()
                perdida.backward()
                opt.step()
                suma += float(perdida.detach()) * nb
                n_vistos += nb
            registro = {"perdida": suma / n_vistos}
            historial["perdida"].append(registro["perdida"])
            if cb:
                cb(epoca, registro)
        return historial

    @torch.no_grad()
    def muestrear(self, n, y=None, pasos=None, guia=1.0,
                  guardar_trayectoria=False):
        """Muestreo inverso (DDIM determinista sobre un subcalendario).

        pasos: cuántos pasos usar (por defecto, todos los del
        entrenamiento). Menos pasos = más rápido, menos fiel.
        guia: escala de guía sin clasificador. 1.0 = condicional puro;
        >1 = más adherencia a la condición, menos diversidad. Sólo tiene
        efecto si se pasa ``y``.
        guardar_trayectoria=True devuelve (muestras, trayectoria) donde
        trayectoria es la lista de estados intermedios (para
        ``graficas.trayectoria_2d``).
        """
        pasos = pasos or self.pasos
        forma = _forma_muestra(self.meta)
        x = torch.randn(n, *forma, device=self.dispositivo)
        y_cond = None if y is None else self._y_o_nula(y, n)
        y_nula = self._y_o_nula(None, n)

        indices = torch.linspace(self.pasos - 1, 0, pasos).long()
        trayectoria = [x.cpu().clone()]
        for i, t_i in enumerate(indices):
            t = torch.full((n,), int(t_i), device=self.dispositivo)
            if y_cond is None:
                ruido_pred = self.red(x, t, y_nula)
            else:
                r_cond = self.red(x, t, y_cond)
                if guia == 1.0:
                    ruido_pred = r_cond
                else:
                    r_nulo = self.red(x, t, y_nula)
                    ruido_pred = r_nulo + guia * (r_cond - r_nulo)

            ac = self.alfas_cum[t_i]
            x0_pred = (x - (1 - ac).sqrt() * ruido_pred) / ac.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)
            # ruido implícito CONSISTENTE con el x0 recortado: sin esto,
            # un ruido sobreestimado se retroalimenta paso a paso y el
            # muestreo diverge (verificado: |x| → NaN en ~140 pasos).
            ruido_pred = (x - ac.sqrt() * x0_pred) / (1 - ac).sqrt()
            if i + 1 < len(indices):
                ac_sig = self.alfas_cum[indices[i + 1]]
                x = ac_sig.sqrt() * x0_pred + (1 - ac_sig).sqrt() * ruido_pred
            else:
                x = x0_pred
            if guardar_trayectoria:
                trayectoria.append(x.cpu().clone())

        if guardar_trayectoria:
            return x.cpu(), trayectoria
        return x.cpu()
