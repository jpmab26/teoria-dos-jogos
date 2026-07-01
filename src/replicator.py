"""
Dinamica de replicador do jogo evolutivo assimetrico entre defensores e
atacantes de um NCS (Networked Control System). Ver o README para a
explicacao completa do jogo (jogadores, acoes, bimatriz de payoffs) e da
metodologia de simulacao; este modulo implementa apenas a matematica.

Duas populacoes, cada uma com duas estrategias puras:
  - Populacao D (defensores/operadores do NCS): N = manter o controlador
    unico (nao comutante); S = adotar o controlador comutante aleatorio.
  - Populacao A (atacantes MitM): C = lancar o ataque covert/dependente de
    modelo (Identificacao Passiva + Injecao de Dados); Ø = abster-se.

x = fracao de D jogando S; y = fracao de A jogando C. A dinamica de
replicador (taxa de crescimento de cada estrategia proporcional a vantagem
do seu payoff sobre a media da populacao) reduz, para um jogo 2x2, a:

    xdot = x(1-x) * [ y*L*(1-g) - sigma ]
    ydot = y(1-y) * [ (V-kappa) - x*((1-g)*V + p*delta) ]

Os limiares abaixo sao os pontos em que o colchete de cada equacao muda de
sinal -- ou seja, a fracao da populacao oponente acima/abaixo da qual a
melhor resposta de cada populacao se inverte:
    y_star = sigma / (L*(1-g))          (limiar de ataques que justifica S)
    x_star = (V-kappa) / ((1-g)*V + p*delta)   (limiar de defesa que dissuade C)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class GameParams:
    V: float
    L: float
    g: float
    sigma: float
    kappa: float
    p: float
    delta: float

    def as_dict(self):
        return dict(V=self.V, L=self.L, g=self.g, sigma=self.sigma,
                    kappa=self.kappa, p=self.p, delta=self.delta)


def y_star(params: GameParams) -> float:
    return params.sigma / (params.L * (1 - params.g))


def x_star(params: GameParams) -> float:
    return (params.V - params.kappa) / ((1 - params.g) * params.V + params.p * params.delta)


def replicator_rhs(t, state, params: GameParams):
    x, y = state
    xdot = x * (1 - x) * (y * params.L * (1 - params.g) - params.sigma)
    ydot = y * (1 - y) * ((params.V - params.kappa) - x * ((1 - params.g) * params.V + params.p * params.delta))
    return [xdot, ydot]


def jacobian_interior_analytic(params: GameParams):
    """Jacobiano analitico no ponto interior (x*, y*) (ver derivacao no
    README / resultados.md): a diagonal principal se anula identicamente
    porque f(y*)=0 e h(x*)=0 por definicao dos limiares, restando uma
    matriz anti-simetrica cujos autovalores sao sempre puramente
    imaginarios (CENTRO), confirmando analiticamente o Teorema 9.8 de
    Gintis (1999) para este jogo 2x2.
    """
    xs, ys = x_star(params), y_star(params)
    if not (0 < xs < 1 and 0 < ys < 1):
        return None, None, None
    a12 = xs * (1 - xs) * params.L * (1 - params.g)
    a21 = -ys * (1 - ys) * ((1 - params.g) * params.V + params.p * params.delta)
    J = np.array([[0.0, a12], [a21, 0.0]])
    eigvals = np.linalg.eigvals(J)
    return J, eigvals, (xs, ys)


def jacobian_interior_numeric(params: GameParams, eps: float = 1e-6):
    xs, ys = x_star(params), y_star(params)
    if not (0 < xs < 1 and 0 < ys < 1):
        return None, None
    J = np.zeros((2, 2))
    for j, dvec in enumerate([(eps, 0.0), (0.0, eps)]):
        f_plus = replicator_rhs(0, (xs + dvec[0], ys + dvec[1]), params)
        f_minus = replicator_rhs(0, (xs - dvec[0], ys - dvec[1]), params)
        J[:, j] = (np.array(f_plus) - np.array(f_minus)) / (2 * eps)
    eigvals = np.linalg.eigvals(J)
    return J, eigvals


def classify_vertices(params: GameParams):
    """Para cada vertice (x,y) in {0,1}^2 da bimatriz, verifica se e um
    equilibrio de Nash estrito (nenhum dos dois jogadores ganha ao desviar
    unilateralmente), usando diretamente os limiares y_star, x_star (validos
    independente de estarem ou nao dentro do intervalo aberto (0,1))."""
    xs, ys = x_star(params), y_star(params)
    V, kappa = params.V, params.kappa

    vertices = {
        "(N,Ø)": dict(x=0, y=0, is_nash=not (V > kappa)),
        "(N,C)": dict(x=0, y=1, is_nash=(ys >= 1) and (V > kappa)),
        "(S,Ø)": dict(x=1, y=0, is_nash=(params.sigma <= 0)),
        "(S,C)": dict(x=1, y=1, is_nash=(ys < 1) and (xs >= 1)),
    }
    return vertices, dict(x_star=xs, y_star=ys)


def simulate_trajectories(params: GameParams, initial_conditions, t_span=(0, 500), n_points=5000):
    trajectories = []
    t_eval = np.linspace(*t_span, n_points)
    for x0, y0 in initial_conditions:
        sol = solve_ivp(
            replicator_rhs, t_span, [x0, y0], args=(params,), t_eval=t_eval,
            rtol=1e-9, atol=1e-11, method="RK45",
        )
        trajectories.append(dict(x0=x0, y0=y0, t=sol.t, x=sol.y[0], y=sol.y[1]))
    return trajectories


def time_average_ergodicity_check(params: GameParams, x0=0.6, y0=0.6, t_span=(0, 2000), n_points=200000):
    """Verifica a propriedade ergodica do Teorema 9.8 de Gintis (1999):
    a media temporal de uma orbita fechada deve coincidir com (x*, y*)."""
    sol = solve_ivp(
        replicator_rhs, t_span, [x0, y0], args=(params,),
        t_eval=np.linspace(*t_span, n_points), rtol=1e-10, atol=1e-12, method="RK45",
    )
    x_mean = float(np.trapezoid(sol.y[0], sol.t) / (t_span[1] - t_span[0]))
    y_mean = float(np.trapezoid(sol.y[1], sol.t) / (t_span[1] - t_span[0]))
    return dict(x_time_avg=x_mean, y_time_avg=y_mean, x_star=x_star(params), y_star=y_star(params))


def estimate_orbit_period(params: GameParams, x0=0.6, y0=0.6, t_span=(0, 200), n_points=20000):
    """Estima o periodo da orbita fechada a partir dos cruzamentos por y* na
    direcao ascendente (proxy simples e robusto para orbitas aproximadamente
    fechadas)."""
    sol = solve_ivp(
        replicator_rhs, t_span, [x0, y0], args=(params,),
        t_eval=np.linspace(*t_span, n_points), rtol=1e-10, atol=1e-12, method="RK45",
    )
    ys = y_star(params)
    y = sol.y[1]
    t = sol.t
    crossings = []
    for i in range(1, len(y)):
        if y[i - 1] < ys <= y[i]:
            frac = (ys - y[i - 1]) / (y[i] - y[i - 1])
            crossings.append(t[i - 1] + frac * (t[i] - t[i - 1]))
    if len(crossings) < 2:
        return None
    periods = np.diff(crossings)
    return dict(mean_period=float(np.mean(periods)), std_period=float(np.std(periods)), n_cycles=len(periods))


def sensitivity_sweep(base_params: GameParams, param_name: str, values):
    xs_list, ys_list = [], []
    for v in values:
        kwargs = base_params.as_dict()
        kwargs[param_name] = v
        p = GameParams(**kwargs)
        xs_list.append(x_star(p))
        ys_list.append(y_star(p))
    return np.array(xs_list), np.array(ys_list)
