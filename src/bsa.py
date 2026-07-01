"""
Implementacao simplificada do Backtracking Search Algorithm (BSA, Civicioglu
2013) usada pelo ataque de Identificacao Passiva (Secao 3.1.2 do artigo).

Esta e a parte OPCIONAL/SECUNDARIA do enunciado (item 4 da Parte A): aqui
identificamos apenas o CONTROLADOR (problema 2D, forma C(z)=(a z - b)/(z-1)),
que e a comparacao mais informativa do artigo (Fig. 9: dispersao das
coordenadas estimadas sob o controlador comutante vs. concentracao sob o
controlador unico). A identificacao da planta (problema 4D, Fig. 10 parcial)
NAO e executada em escala completa (100 repeticoes x 800 iteracoes cada) por
custo computacional desproporcional ao foco deste trabalho (a calibracao do
jogo evolutivo, Partes B e C) -- fica documentada aqui como extensao futura,
conforme autorizado pelo enunciado ("se custar muito tempo, documente como
extensao e siga").

Simplificacoes assumidas em relacao ao BSA "canonico" de Civicioglu (2013),
documentadas para transparencia:
  - O fator de mutacao Gamma ~ N(0,1) e sorteado uma unica vez por iteracao
    (escalar global), conforme a Eq. (4) do artigo, tal qual escrita.
  - O operador de crossover usa mix-rate=1.0 por padrao (todas as dimensoes
    do indivíduo sao atualizadas pelo mutante a cada iteracao) -- uma
    simplificacao do crossover combinatorio original do BSA, que nao muda a
    natureza do algoritmo (historico-guiado, selecao gulosa) mas reduz o
    espaco de hiperparametros a calibrar.
"""
from __future__ import annotations

import numpy as np


def _controller_predict(params: np.ndarray, e_signal: np.ndarray) -> np.ndarray:
    """params: (pop, 2) = (a, b) para C(z) = (a z + b)/(z - 1) (convencao de
    sinal adotada em src/ncs_model.py -- ver nota "RESOLUCAO DE AMBIGUIDADE
    DE SINAL" la).
    Recursao: u(k) = u(k-1) + a*e(k) + b*e(k-1).
    Retorna u_hat com shape (pop, K)."""
    pop = params.shape[0]
    K = e_signal.shape[0]
    a = params[:, 0]
    b = params[:, 1]
    u_hat = np.zeros((pop, K))
    e_prev = 0.0
    u_prev = np.zeros(pop)
    for k in range(K):
        e_k = e_signal[k]
        u_k = u_prev + a * e_k + b * e_prev
        u_hat[:, k] = u_k
        u_prev = u_k
        e_prev = e_k
    return u_hat


def _plant_predict(params: np.ndarray, i_signal: np.ndarray) -> np.ndarray:
    """params: (pop, 4) = (g1, g2, g3, g4) para
    G(z) = (g1 z + g2) / (z^2 + g3 z + g4) (convencao de sinal adotada em
    src/ncs_model.py).
    Recursao: y(k) = -g3*y(k-1) - g4*y(k-2) + g1*i(k-1) + g2*i(k-2).
    Retorna y_hat com shape (pop, K)."""
    pop = params.shape[0]
    K = i_signal.shape[0]
    g1, g2, g3, g4 = (params[:, j] for j in range(4))
    y_hat = np.zeros((pop, K))
    for k in range(2, K):
        y_hat[:, k] = (
            -g3 * y_hat[:, k - 1] - g4 * y_hat[:, k - 2]
            + g1 * i_signal[k - 1] + g2 * i_signal[k - 2]
        )
    return y_hat


def _fitness(predict_fn, params, i_signal, o_signal):
    pred = predict_fn(params, i_signal)
    return np.mean((o_signal[None, :] - pred) ** 2, axis=1)


def bsa_identify(
    i_signal: np.ndarray,
    o_signal: np.ndarray,
    dim: int,
    predict_fn,
    pop_size: int = 100,
    iters: int = 600,
    bounds=(-10.0, 10.0),
    eta: float = 1.0,
    mix_rate: float = 1.0,
    rng: np.random.Generator | None = None,
):
    """BSA generico (ver simplificacoes no docstring do modulo).

    Retorna (best_params, best_fitness, final_population, final_fitness).
    """
    rng = rng or np.random.default_rng()
    lo, hi = bounds

    P = rng.uniform(lo, hi, size=(pop_size, dim))
    P_hist = rng.uniform(lo, hi, size=(pop_size, dim))
    fit = _fitness(predict_fn, P, i_signal, o_signal)

    best_idx = np.argmin(fit)
    best_params = P[best_idx].copy()
    best_fit = fit[best_idx]

    for _ in range(iters):
        # Selecao-I
        if rng.random() < rng.random():
            P_hist = P.copy()
        perm = rng.permutation(pop_size)
        P_hist = P_hist[perm]

        # Mutacao (Eq. 4 do artigo)
        gamma = rng.standard_normal()
        P_mod = P + eta * gamma * (P_hist - P)

        # Crossover (simplificado, ver docstring do modulo)
        mask = rng.random((pop_size, dim)) < mix_rate
        # garante ao menos 1 dimensao alterada por individuo
        no_change = ~mask.any(axis=1)
        if np.any(no_change):
            cols = rng.integers(0, dim, size=no_change.sum())
            mask[no_change, cols] = True
        P_new = np.where(mask, P_mod, P)
        P_new = np.clip(P_new, lo, hi)

        # Selecao-II (gulosa)
        fit_new = _fitness(predict_fn, P_new, i_signal, o_signal)
        improve = fit_new < fit
        P[improve] = P_new[improve]
        fit[improve] = fit_new[improve]

        gen_best_idx = np.argmin(fit)
        if fit[gen_best_idx] < best_fit:
            best_fit = fit[gen_best_idx]
            best_params = P[gen_best_idx].copy()

    return best_params, float(best_fit), P, fit


def identify_controller(e_signal, u_signal, **kwargs):
    return bsa_identify(e_signal, u_signal, dim=2, predict_fn=_controller_predict, **kwargs)


def identify_plant(i_signal, o_signal, **kwargs):
    return bsa_identify(i_signal, o_signal, dim=4, predict_fn=_plant_predict, **kwargs)


def run_part_a4(n_trials: int = 30, pop_size: int = 100, iters: int = 600, seed: int = 0):
    """Reproduz (em escala reduzida) o experimento da Fig. 9 do artigo:
    identificacao do controlador (2D) via BSA, comparando a dispersao das
    coordenadas estimadas sob o controlador NAO comutante (N) vs. o
    controlador COMUTANTE (S), ambos com janela de monitoramento T=2s
    (K=100 amostras).

    Reducao de escala em relacao ao artigo (100 -> `n_trials` repeticoes):
    ver docstring do modulo. `pop_size`/`iters` sao mantidos identicos ao
    artigo (100 / 600) pois o custo por execucao e pequeno (problema 2D).
    """
    from .ncs_model import (
        FS, plant_block, controller1_block, controller2_block,
        simulate_non_switching, simulate_switching_monte_carlo,
        C1_COEFFS, C2_COEFFS,
    )

    n_steps = int(2 * FS)  # T = 2s, K = 100 amostras
    plant = plant_block()
    c1 = controller1_block()
    c2 = controller2_block()

    rng = np.random.default_rng(seed)

    # --- Controlador NAO comutante: sinal capturado e o MESMO em todas as
    # repeticoes (sistema deterministico); toda a dispersao observada vem da
    # estocasticidade do proprio BSA.
    y_n, u_n = simulate_non_switching(plant, c1, n_steps)
    e_n = 1.0 - y_n

    results_n = []
    for _ in range(n_trials):
        best, best_fit, _, _ = identify_controller(
            e_n, u_n, pop_size=pop_size, iters=iters, rng=rng
        )
        results_n.append((best[0], best[1], best_fit))

    # --- Controlador comutante: cada repeticao usa uma realizacao distinta
    # da comutacao aleatoria (o atacante captura uma janela de 2s a cada
    # tentativa de ataque).
    results_s = []
    for trial in range(n_trials):
        y_s, u_s = simulate_switching_monte_carlo(
            plant, c1, c2, n_steps, n_sims=1, seed=int(rng.integers(0, 2**31 - 1)),
            record_u=True,
        )
        e_s = 1.0 - y_s[:, 0]
        best, best_fit, _, _ = identify_controller(
            e_s, u_s[:, 0], pop_size=pop_size, iters=iters, rng=rng
        )
        results_s.append((best[0], best[1], best_fit))

    results_n = np.array(results_n)  # (n_trials, 3) = a, b, fitness
    results_s = np.array(results_s)

    return dict(
        n_trials=n_trials, pop_size=pop_size, iters=iters,
        actual_a=C1_COEFFS["c11"], actual_b=C1_COEFFS["c21"],
        c2_actual_a=C2_COEFFS["c12"], c2_actual_b=C2_COEFFS["c22"],
        results_non_switching=results_n,
        results_switching=results_s,
        fitness_mean_n=float(results_n[:, 2].mean()),
        fitness_std_n=float(results_n[:, 2].std()),
        fitness_mean_s=float(results_s[:, 2].mean()),
        fitness_std_s=float(results_s[:, 2].std()),
    )
