import argparse
import json
import math
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LB, UB = -1.0, 2.0


def aptitud(x):
    return x * np.sin(10.0 * np.pi * x) + 1.0


def decodificar(pob, nbits):
    pesos = 2 ** np.arange(nbits - 1, -1, -1, dtype=np.int64)
    enteros = pob.astype(np.int64) @ pesos
    return LB + enteros * (UB - LB) / (2 ** nbits - 1)


def seleccion_torneo(pob, fit, rng, k=3):
    n = len(pob)
    cand = rng.integers(0, n, size=(n, k))
    ganadores = cand[np.arange(n), np.argmax(fit[cand], axis=1)]
    return pob[ganadores].copy()


def seleccion_ruleta(pob, fit, rng, k=None):
    f = fit - fit.min() + 1e-9
    p = f / f.sum()
    idx = rng.choice(len(pob), size=len(pob), p=p)
    return pob[idx].copy()


def cruce(p1, p2, tipo, rng):
    L = len(p1)
    if tipo == "un_punto":
        c = rng.integers(1, L)
        h1 = np.concatenate([p1[:c], p2[c:]])
        h2 = np.concatenate([p2[:c], p1[c:]])
    elif tipo == "dos_puntos":
        a, b = sorted(rng.choice(np.arange(1, L), size=2, replace=False))
        h1, h2 = p1.copy(), p2.copy()
        h1[a:b], h2[a:b] = p2[a:b], p1[a:b]
    elif tipo == "uniforme":
        m = rng.random(L) < 0.5
        h1 = np.where(m, p1, p2)
        h2 = np.where(m, p2, p1)
    else:
        raise ValueError(f"Tipo de cruce desconocido: {tipo}")
    return h1, h2


def mutacion_bitflip(pob, pm, rng):
    mascara = rng.random(pob.shape) < pm
    return np.where(mascara, 1 - pob, pob)


def diversidad(pob):
    p = pob.mean(axis=0)
    return float(np.mean(4.0 * p * (1.0 - p)))


def ejecutar(cfg):
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg["semilla"])
    N, G = cfg["tamano_poblacion"], cfg["generaciones"]
    nbits = cfg["bits"]
    sel = seleccion_torneo if cfg["seleccion"] == "torneo" else seleccion_ruleta
    n_elite = cfg.get("elitismo", 1)

    pob = rng.integers(0, 2, size=(N, nbits))
    hist_mejor, hist_media, hist_div = [], [], []
    mejor_global, x_global, ind_global = -np.inf, None, None

    for g in range(G + 1):
        x = decodificar(pob, nbits)
        fit = aptitud(x)
        i_best = int(np.argmax(fit))
        if fit[i_best] > mejor_global:
            mejor_global, x_global, ind_global = fit[i_best], x[i_best], pob[i_best].copy()
        hist_mejor.append(mejor_global)
        hist_media.append(float(fit.mean()))
        hist_div.append(diversidad(pob))
        if g == G:
            break

        elite = pob[np.argsort(fit)[-n_elite:]].copy() if n_elite > 0 else None

        padres = sel(pob, fit, rng, cfg.get("k_torneo", 3))

        hijos = padres.copy()
        for i in range(0, N - 1, 2):
            if rng.random() < cfg["prob_cruce"]:
                hijos[i], hijos[i + 1] = cruce(padres[i], padres[i + 1], cfg["tipo_cruce"], rng)

        hijos = mutacion_bitflip(hijos, cfg["prob_mutacion"], rng)

        if n_elite > 0:
            hijos[:n_elite] = elite
        pob = hijos

    hist_mejor = np.array(hist_mejor)
    gen_estab = int(np.argmax(hist_mejor >= hist_mejor[-1] - 1e-12))
    return {
        "mejor_x": float(x_global),
        "mejor_f": float(mejor_global),
        "mejor_bits": "".join(map(str, ind_global)),
        "hist_mejor": hist_mejor,
        "hist_media": np.array(hist_media),
        "hist_div": np.array(hist_div),
        "gen_estab": gen_estab,
        "tiempo": time.perf_counter() - t0,
    }


def fuerza_bruta(n_puntos=3_000_001):
    xs = np.linspace(LB, UB, n_puntos)
    fs = aptitud(xs)
    i = int(np.argmax(fs))
    return float(xs[i]), float(fs[i])


def graficar(res, ruta, titulo=""):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(res["hist_mejor"], label="Mejor aptitud", lw=2)
    ax.plot(res["hist_media"], label="Aptitud media", lw=1, ls="--")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Aptitud f(x)")
    ax.set_title(titulo or "Convergencia del algoritmo genético")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def reporte(cfg, res):
    x_bf, f_bf = fuerza_bruta()
    err = f_bf - res["mejor_f"]
    lineas = [
        "=" * 60,
        "REPORTE DEL MEJOR INDIVIDUO",
        "=" * 60,
        f"Cromosoma ({cfg['bits']} bits): {res['mejor_bits']}",
        f"x encontrado          : {res['mejor_x']:.6f}",
        f"f(x) encontrado       : {res['mejor_f']:.6f}",
        f"Generación estabilizó : {res['gen_estab']}",
        f"Tiempo de ejecución   : {res['tiempo']:.3f} s",
        "-" * 60,
        "VALIDACIÓN (fuerza bruta, 3,000,001 puntos)",
        f"x óptimo              : {x_bf:.6f}",
        f"f(x) óptimo           : {f_bf:.6f}",
        f"Error absoluto        : {err:.2e}",
        "=" * 60,
    ]
    return "\n".join(lineas)


def cargar_config(ruta):
    with open(ruta, "r", encoding="utf-8") as fh:
        return json.load(fh)


def menu(cfg):
    campos = [
        ("tamano_poblacion", int), ("generaciones", int), ("prob_cruce", float),
        ("prob_mutacion", float), ("semilla", int), ("seleccion", str),
        ("tipo_cruce", str), ("k_torneo", int), ("elitismo", int), ("bits", int),
    ]
    while True:
        print("\n--- MENÚ DEL ALGORITMO GENÉTICO ---")
        for i, (k, _) in enumerate(campos, 1):
            print(f"{i:2d}) {k:18s} = {cfg[k]}")
        print(" E) Ejecutar      S) Salir")
        op = input("Opción: ").strip().lower()
        if op == "e":
            return cfg
        if op == "s":
            raise SystemExit(0)
        if op.isdigit() and 1 <= int(op) <= len(campos):
            k, tipo = campos[int(op) - 1]
            try:
                cfg[k] = tipo(input(f"Nuevo valor para {k}: ").strip())
            except ValueError:
                print("Valor inválido.")


def main():
    ap = argparse.ArgumentParser(description="AG para max x*sin(10*pi*x)+1 en [-1,2]")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--menu", action="store_true", help="abrir menú interactivo")
    ap.add_argument("--salida", default="convergencia.png")
    args = ap.parse_args()

    cfg = cargar_config(args.config)
    if args.menu:
        cfg = menu(cfg)
    res = ejecutar(cfg)
    graficar(res, args.salida)
    print(reporte(cfg, res))
    print(f"Gráfica de convergencia guardada en: {args.salida}")


if __name__ == "__main__":
    main()