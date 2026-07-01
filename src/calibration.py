"""
Traduz os resultados de controle (Parte A) em payoffs do jogo evolutivo
(Parte C). Calcula os 4 parametros do jogo que tem contrapartida numerica
direta na dinamica da planta -- V, L, g, sigma -- a partir de duas fontes,
reportadas lado a lado por transparencia:

  - "artigo": valores tirados diretamente da literatura de origem sobre o
    NCS e a contramedida do controlador comutante, usados como calibracao
    PRIMARIA/de referencia.
  - "replicado": valores obtidos da nossa propria replicacao computacional
    (Parte A), que reproduz os numeros de referencia com erro tipicamente
    abaixo de 5% (ver resultados.md para a tabela completa de validacao).

As duas fontes concordarem de perto e uma checagem de consistencia: reforca
que os parametros calibrados do jogo (e, em particular, o achado de
"knife-edge" discutido no README) nao sao um artefato de uma escolha de
modelagem especifica, mas sim uma consequencia robusta dos numeros de
controle em si.

Os outros 3 parametros do jogo -- kappa (custo do ataque), p (probabilidade
de deteccao) e delta (penalidade por deteccao) -- nao tem contrapartida
numerica direta na literatura de controle consultada; sao tratados como
parametros de projeto, com um cenario-base ilustrativo e uma varredura de
sensibilidade ampla (Parte C / `replicator.py`).
"""
from __future__ import annotations

from dataclasses import asdict

from .replicator import GameParams


def calibrate_from_article():
    """Calibracao primaria, com base nos numeros publicados no artigo:
      - overshoot pretendido = 50%, overshoot real contra N = 48.90%
      - overshoot real contra S (com M(z) redesenhado) = 10.12%
      - settling time N = 2.4s, settling time medio S = 4.2827s
    """
    v_target_overshoot = 50.0
    l_actual_overshoot_n = 48.90
    overshoot_s = 10.12
    ts_n = 2.4
    ts_s_mean = 4.2827

    V = 1.0
    L = l_actual_overshoot_n / v_target_overshoot
    g = overshoot_s / v_target_overshoot
    sigma = (ts_s_mean - ts_n) / ts_n  # aumento RELATIVO do settling time

    return dict(V=V, L=L, g=g, sigma=sigma, source="artigo",
                raw=dict(v_target_overshoot=v_target_overshoot,
                         l_actual_overshoot_n=l_actual_overshoot_n,
                         overshoot_s=overshoot_s, ts_n=ts_n, ts_s_mean=ts_s_mean))


def calibrate_from_replication(part_a3_results: dict, ncs_control_results: dict):
    """Calibracao secundaria (cross-check), com base na nossa propria
    replicacao computacional (Parte A)."""
    ns = part_a3_results["non_switching"]
    target = 50.0
    L = ns["actual_overshoot_pct"] / target
    V = 1.0

    ts_n = ncs_control_results["non_switching"]["settling_time_s"]
    ts_s_mean = ncs_control_results["switching"]["settling_time_mean_s"]
    sigma = (ts_s_mean - ts_n) / ts_n

    ov_s_given_ko = ncs_control_results["switching_attack_given_ko"]["overshoot_max_pct"]
    g = ov_s_given_ko / target

    return dict(V=V, L=L, g=g, sigma=sigma, source="replicado",
                raw=dict(ts_n=ts_n, ts_s_mean=ts_s_mean,
                         actual_overshoot_n=ns["actual_overshoot_pct"],
                         overshoot_s_given_ko=ov_s_given_ko))


def build_baseline_game_params(calib: dict, kappa=0.1, p=0.3, delta=1.0) -> GameParams:
    """kappa, p, delta nao tem contrapartida numerica direta na literatura
    de controle consultada; aqui fixamos um cenario-base ilustrativo
    (kappa=0.1V: ataque barato; p=0.3: deteccao moderada; delta=1.0V:
    penalidade da mesma ordem do valor do ataque) e a analise de
    sensibilidade completa (Parte C) varre os tres em faixas amplas."""
    return GameParams(
        V=calib["V"], L=calib["L"], g=calib["g"], sigma=calib["sigma"],
        kappa=kappa, p=p, delta=delta,
    )


def build_pedagogical_game_params(calib: dict, sigma_scale: float = 0.5, kappa=0.1, p=0.3, delta=1.0) -> GameParams:
    """Cenario PEDAGOGICO (nao calibrado): com os parametros calibrados
    reais (fonte artigo ou replicado), o limiar y* = sigma/(L(1-g)) fica
    muito proximo de -- ou acima de -- 1 (ver knife_edge_note em
    calibrate_from_article/build_baseline_game_params e a discussao no
    README), de modo que o ponto interior do jogo deixa de existir dentro
    de (0,1)^2 nesses cenarios. Para ilustrar o comportamento generico de
    centro/orbita fechada da dinamica de replicador -- o resultado teorico
    central da Parte C --, este cenario reduz sigma artificialmente (por um
    fator `sigma_scale`), mantendo V, L, g, kappa, p, delta calibrados/
    ilustrativos inalterados. NAO deve ser interpretado como uma segunda
    calibracao empirica, apenas como um cenario didatico com ponto interior
    genuino, usado exclusivamente para gerar as Figs. C1-C2 (retrato de fase
    e series temporais)."""
    return GameParams(
        V=calib["V"], L=calib["L"], g=calib["g"], sigma=calib["sigma"] * sigma_scale,
        kappa=kappa, p=p, delta=delta,
    )
