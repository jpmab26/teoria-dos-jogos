"""
Realizacao em espaco de estados e simulacao em malha fechada de um Sistema de
Controle em Rede (NCS) formado por uma planta de motor DC de 2a ordem e um
controlador PI, com um segundo controlador PI alternativo usado para compor
um controlador comutante aleatorio. Os valores numericos dos coeficientes
(g1..g4, c1,1, c2,1, c1,2, c2,2) sao os publicados na literatura de origem
sobre mitigacao de ataques de identificacao passiva em NCS (motor DC,
controlador PI, taxa de amostragem 50 amostras/s).

RESOLUCAO DE AMBIGUIDADE DE SINAL
----------------------------------
A formula tipograficamente publicada para a planta e o controlador nao
comutante deixa ambiguo, apos a extracao do texto original, se os termos de
2o coeficiente (g3 no denominador da planta; c2,1 no numerador do
controlador) entram com o sinal literal ou com o sinal invertido -- um
problema comum quando expressoes com subscritos e sinais de menos sao
extraidas de PDF. Em vez de adotar a leitura literal sem verificacao,
testamos exaustivamente as 8 combinacoes de sinal possiveis (2 para o
denominador da planta x 4 para o numerador do controlador) e avaliamos cada
uma por tres criterios independentes:
  1. Estabilidade em malha fechada com os ganhos do controlador publicados
     (o artigo afirma explicitamente que esse par planta/controlador e
     estavel, com tempo de acomodacao de 2.4s);
  2. Plausibilidade fisica dos polos da planta (um motor DC, sendo um
     sistema de 2a ordem sem zeros de fase nao-minima conhecidos, tem
     tipicamente polos reais POSITIVOS quando discretizado por ZOH a partir
     de polos continuos reais estaveis -- polos negativos implicariam uma
     oscilacao a cada amostra, fisicamente atipica para velocidade de um
     motor);
  3. Forma qualitativa da resposta ao ataque de ganho: o artigo mostra (em
     suas figuras) um unico sobressinal suave ao inserir o ganho de ataque
     Ko, nao um batimento/ringing de alta frequencia.
Uma unica combinacao satisfaz os tres criterios simultaneamente: polos da
planta positivos (denominador `z^2 + g3*z + g4`, plugando o valor de g3
diretamente) combinados com o numerador do controlador C1(z) SEM negar o
segundo coeficiente (`c1,1*z + c2,1`). Essa e a convencao adotada em
`plant_block`/`controller1_block` abaixo (ver as notas de cada funcao). Com
ela, o tempo de acomodacao replicado (2.28s) fica a ~5% do publicado (2.4s),
contra ~52% de erro da leitura literal alternativa -- ver `resultados.md`
para a tabela de validacao completa e a comparacao numerica entre as duas
convencoes.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

# ---------------------------------------------------------------------------
# Parametros publicados no artigo (Secao 3.3, Eqs. 5 e 6; Secao 5, Eq. 7)
# ---------------------------------------------------------------------------
FS = 50.0          # taxa de amostragem (amostras/s)
TS = 1.0 / FS       # periodo de amostragem (s)

# Planta G(z) = (g1 z + g2) / (z^2 - g3 z + g4)
G_COEFFS = dict(g1=0.3379, g2=0.2793, g3=-1.5462, g4=0.5646)

# Controlador PI nao comutante C1(z) = (c11 z - c21) / (z - 1)
C1_COEFFS = dict(c11=0.1701, c21=-0.1673)

# Segundo controlador (usado no controlador comutante) C2(z) = (c12 z + c22) / (z - 1)
C2_COEFFS = dict(c12=0.001, c22=0.0002)

# Coeficientes MEDIOS estimados pelo ataque de Identificacao Passiva sob o
# controlador NAO comutante, com 0% de perda de amostras (Tabela 2 do artigo).
# Usamos estes valores -- publicados diretamente no artigo -- para projetar a
# funcao de ataque M(z) sem precisar re-executar o BSA (que e tratado como
# extensao opcional secundaria neste projeto, Secao 4 item 4 do enunciado).
EST_COEFFS_0PCT = dict(
    g1=0.32793, g2=0.29652, g3=-1.54121, g4=0.55983,
    c11=0.16991, c21=-0.16712,
)

# Parametros da PDF de dwell-time (Secao 5, paragrafo 3)
A_DWELL = 40   # amostras (0.8 s)
B_DWELL = 60   # amostras (1.2 s)


class LTIBlock:
    """Realizacao em espaco de estados (forma canonica controlavel) de uma
    funcao de transferencia discreta propria, obtida via scipy.signal.tf2ss.
    """

    def __init__(self, num, den, name=""):
        num = np.atleast_1d(np.asarray(num, dtype=float))
        den = np.atleast_1d(np.asarray(den, dtype=float))
        A, B, C, D = signal.tf2ss(num, den)
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float).reshape(-1, 1)
        self.C = np.asarray(C, dtype=float).reshape(1, -1)
        self.D = float(np.asarray(D, dtype=float).reshape(-1)[0])
        self.n_states = self.A.shape[0]
        self.name = name
        poles = np.linalg.eigvals(self.A) if self.n_states else np.array([])
        self.poles = poles

    def is_stable(self):
        if self.n_states == 0:
            return True
        return bool(np.all(np.abs(self.poles) < 1.0 + 1e-9))


def plant_block(coeffs=None, gain: float = 1.0):
    """Bloco da planta G(z). `gain` multiplica o numerador (equivalente a
    inserir um ganho M(z)=Ko em serie no caminho direto, usado para simular
    o ataque SD-Controlled Data Injection, Secao 3.2 do artigo).

    Convencao de sinal do denominador: ver a nota "RESOLUCAO DE AMBIGUIDADE
    DE SINAL" no topo deste modulo. Usamos `den = [1, g3, g4]` (plugando o
    valor de g3 diretamente, sem negar), que da polos reais POSITIVOS
    (~0.955, ~0.591) -- fisicamente plausiveis para um motor DC -- em vez de
    `den = [1, -g3, g4]` (leitura literal da Eq. 6), que da polos negativos
    e e instavel/qualitativamente errada sob ganho de ataque.
    """
    c = coeffs or G_COEFFS
    num = [gain * c["g1"], gain * c["g2"]]
    den = [1.0, c["g3"], c["g4"]]
    return LTIBlock(num, den, name="G(z)")


def controller1_block(coeffs=None):
    """Bloco do controlador PI C1(z). Convencao de sinal do numerador: ver a
    nota "RESOLUCAO DE AMBIGUIDADE DE SINAL" no topo deste modulo. Usamos
    `num = [c11, c21]` (sem negar c21), consistente com a planta de polos
    positivos acima -- a combinacao [G positivo, C1 com c21 nao negado] e a
    UNICA, dentre as 8 combinacoes de sinal testadas, que e simultaneamente
    (a) estavel em malha fechada, (b) fisicamente plausivel (polos da planta
    positivos) e (c) produz polos de malha fechada complexos conjugados,
    compativeis com a resposta subamortecida "classica" (um unico
    sobressinal, sem batimento de alta frequencia) mostrada nas figuras do
    artigo.
    """
    c = coeffs or C1_COEFFS
    num = [c["c11"], c["c21"]]
    den = [1.0, -1.0]
    return LTIBlock(num, den, name="C1(z)")


def controller2_block(coeffs=None):
    c = coeffs or C2_COEFFS
    num = [c["c12"], c["c22"]]
    den = [1.0, -1.0]
    return LTIBlock(num, den, name="C2(z)")


# ---------------------------------------------------------------------------
# Simulacao vetorizada em malha fechada (realimentacao unitaria, rede ideal:
# sem atraso/perda de pacotes na simulacao de controle -- a perda de amostras
# e usada apenas no ataque de identificacao, Secao 3.1, e nao no laco de
# controle em si).
# ---------------------------------------------------------------------------

def simulate_non_switching(plant: LTIBlock, ctrl: LTIBlock, n_steps: int, r_value: float = 1.0):
    """Simulacao deterministica (1 trajetoria) do NCS com controlador unico.

    Retorna y (n_steps,), u (n_steps,).
    """
    if plant.D != 0.0:
        raise ValueError("Esta implementacao assume D=0 na planta (sem laco algebrico).")
    xp = np.zeros(plant.n_states)
    xc = np.zeros(ctrl.n_states)
    y = np.zeros(n_steps)
    u_hist = np.zeros(n_steps)
    for k in range(n_steps):
        yk = float((plant.C @ xp).reshape(-1)[0])
        y[k] = yk
        e = r_value - yk
        uk = float((ctrl.C @ xc).reshape(-1)[0]) + ctrl.D * e
        u_hist[k] = uk
        xp = plant.A @ xp + plant.B.flatten() * uk
        xc = ctrl.A @ xc + ctrl.B.flatten() * e
    return y, u_hist


def simulate_switching_monte_carlo(
    plant: LTIBlock,
    ctrl0: LTIBlock,
    ctrl1: LTIBlock,
    n_steps: int,
    n_sims: int,
    a: int = A_DWELL,
    b: int = B_DWELL,
    r_value: float = 1.0,
    seed: int | None = None,
    record_active=False,
    record_u=False,
    bumpless: bool = False,
):
    """Simulacao vetorizada de Monte Carlo do NCS com controlador comutante.

    Regra de comutacao: cadeia de Markov com tempo de permanencia (dwell-time)
    sorteado uniformemente em {a, a+1, ..., b} amostras a cada comutacao
    (Secao 4 e Fig. 7-8 do artigo).

    Escolha de modelagem do estado do controlador inativo (nao detalhada na
    fonte original): por padrao (`bumpless=False`), o controlador INATIVO
    tem seu estado interno CONGELADO -- nao e atualizado enquanto nao esta no
    comando da planta -- e retoma exatamente de onde parou quando reativado,
    sem nenhum tratamento especial no instante da troca. Essa e a escolha
    mais simples e a que reproduz de perto os numeros publicados (tempo de
    acomodacao medio ~4.15s, faixa ~2.8-6.4s, sobressinal maximo ~2.9%, ver
    `resultados.md`): a comutacao entre duas funcoes de controle com ganhos
    diferentes injeta uma pequena descontinuidade em u(k) a cada troca, cujo
    efeito acumulado e absorvido pela estabilidade individual dos dois
    subsistemas (ver Secao 4 do texto, sobre "average dwell-time"), sem
    impedir o assentamento da planta.

    Alternativamente (`bumpless=True`), o controlador que ASSUME o controle
    tem seu estado reinicializado por uma transferencia "bumpless" (sem
    solavanco): o novo estado interno e calculado de forma que a saida de
    controle u(k) recalculada coincida com o valor efetivamente aplicado no
    instante anterior a comutacao, eliminando a descontinuidade. Isso e
    tecnicamente mais "seguro" (tecnica padrao em sistemas de controle
    comutados), mas amortece demais o efeito da comutacao neste sistema
    especifico e sub-representa o sobressinal e o tempo de acomodacao
    observados no artigo. Por isso, `bumpless=False` e o padrao adotado.

    Retorna y_hist com shape (n_steps, n_sims) [float32, para economizar
    memoria com n_sims grande].
    """
    if plant.D != 0.0:
        raise ValueError("Esta implementacao assume D=0 na planta (sem laco algebrico).")
    if ctrl0.n_states != ctrl1.n_states:
        raise ValueError("Os dois controladores devem ter o mesmo numero de estados.")

    rng = np.random.default_rng(seed)
    n_p = plant.n_states
    n_c = ctrl0.n_states

    xp = np.zeros((n_sims, n_p))
    xc0 = np.zeros((n_sims, n_c))
    xc1 = np.zeros((n_sims, n_c))
    active = np.zeros(n_sims, dtype=np.int8)  # comeca em C1 (indice 0)
    countdown = rng.integers(a, b + 1, size=n_sims)

    y_hist = np.zeros((n_steps, n_sims), dtype=np.float32)
    active_hist = np.zeros((n_steps, n_sims), dtype=np.int8) if record_active else None
    u_hist = np.zeros((n_steps, n_sims), dtype=np.float32) if record_u else None

    Ap, Bp, Cp = plant.A, plant.B.flatten(), plant.C
    Ac0, Bc0, Cc0, Dc0 = ctrl0.A, ctrl0.B.flatten(), ctrl0.C, ctrl0.D
    Ac1, Bc1, Cc1, Dc1 = ctrl1.A, ctrl1.B.flatten(), ctrl1.C, ctrl1.D
    c0_scalar = float(Cc0.reshape(-1)[0])
    c1_scalar = float(Cc1.reshape(-1)[0])
    if n_c != 1:
        raise NotImplementedError(
            "Reset bumpless implementado apenas para controladores de 1 estado "
            "(caso deste artigo); generalizar via pseudo-inversa se necessario."
        )

    for k in range(n_steps):
        y = (xp @ Cp.T).flatten()
        y_hist[k] = y
        e = r_value - y

        u0 = (xc0 @ Cc0.T).flatten() + Dc0 * e
        u1 = (xc1 @ Cc1.T).flatten() + Dc1 * e
        is_c0 = active == 0
        u = np.where(is_c0, u0, u1)

        if record_active:
            active_hist[k] = active
        if record_u:
            u_hist[k] = u

        xp = xp @ Ap.T + np.outer(u, Bp)

        # Apenas o controlador ATIVO e atualizado (o inativo fica congelado);
        # ver nota de modelagem no docstring desta funcao.
        xc0_new = xc0 @ Ac0.T + np.outer(e, Bc0)
        xc1_new = xc1 @ Ac1.T + np.outer(e, Bc1)
        xc0 = np.where(is_c0[:, None], xc0_new, xc0)
        xc1 = np.where(~is_c0[:, None], xc1_new, xc1)

        countdown -= 1
        switch_mask = countdown <= 0
        n_switch = int(switch_mask.sum())
        if n_switch:
            if bumpless:
                # Reset bumpless do controlador que ASSUME o controle: escolhe
                # o novo estado interno de forma que sua saida, avaliada com o
                # erro corrente e(k), coincida com u(k) efetivamente aplicado
                # agora (evita descontinuidade artificial de u no instante da
                # comutacao).
                switching_to_c1 = switch_mask & is_c0
                switching_to_c0 = switch_mask & ~is_c0
                if np.any(switching_to_c1):
                    reset_val = (u[switching_to_c1] - Dc1 * e[switching_to_c1]) / c1_scalar
                    xc1[switching_to_c1, 0] = reset_val
                if np.any(switching_to_c0):
                    reset_val = (u[switching_to_c0] - Dc0 * e[switching_to_c0]) / c0_scalar
                    xc0[switching_to_c0, 0] = reset_val
            # (modo "frozen": o estado do controlador recem-ativado permanece
            # exatamente como estava na ultima vez em que esteve ativo --
            # nenhuma acao adicional necessaria aqui.)

            active[switch_mask] = 1 - active[switch_mask]
            countdown[switch_mask] = rng.integers(a, b + 1, size=n_switch)

    if not record_active and not record_u:
        return y_hist
    out = [y_hist]
    if record_active:
        out.append(active_hist)
    if record_u:
        out.append(u_hist)
    return tuple(out)


# ---------------------------------------------------------------------------
# Metricas de desempenho de controle
# ---------------------------------------------------------------------------

def settling_time(y, ts=TS, r_value: float = 1.0, tol: float = 0.02):
    """Tempo de acomodacao (criterio de faixa +-tol em torno do valor final).

    Aceita y como array 1D (uma trajetoria) ou 2D (n_steps, n_sims) -- neste
    caso retorna um array de tempos de acomodacao, um por simulacao.
    """
    y = np.asarray(y)
    band = tol * abs(r_value)

    def _single(col):
        outside = np.where(np.abs(col - r_value) > band)[0]
        if outside.size == 0:
            return 0.0
        last_idx = outside[-1]
        if last_idx + 1 >= col.size:
            return np.nan  # nunca acomodou dentro do horizonte simulado
        return (last_idx + 1) * ts

    if y.ndim == 1:
        return _single(y)
    n_steps, n_sims = y.shape
    out = np.empty(n_sims)
    for j in range(n_sims):
        out[j] = _single(y[:, j])
    return out


def overshoot_pct(y, r_value: float = 1.0):
    """Sobressinal percentual em relacao ao valor de referencia r_value."""
    y = np.asarray(y)
    peak = y.max(axis=0)
    return (peak - r_value) / abs(r_value) * 100.0
