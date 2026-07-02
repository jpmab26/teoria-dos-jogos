# Resultados consolidados

Gerado por `main.py`. Todas as figuras estao em `figures/`. Seeds fixas usadas: {"switching_mc": 2026, "switching_attack_mc": 7, "bsa": 1}.

## Tabela de validacao (artigo x replicado)

| Grandeza | Artigo | Replicado | Observacao |
|---|---|---|---|
| Settling time (N) | 2.4 s | 2.4000 s | criterio de banda exp(-4)~=1.83% (ver nota de tolerancia) |
| Overshoot (N) | 0.0% | -0.0065% | |
| Settling time medio (S) | 4.2827 s | 4.2835 s | 100.000 sims Monte Carlo |
| Settling time min (S) | 2.88 s | 2.8800 s | |
| Settling time max (S) | 6.42 s | 6.4600 s | |
| Settling time, IC95% da media (S) | 0.0146 s | 0.0045 s | ver nota sobre IC95 abaixo (desvio-padrao bruto replicado: 0.7274 s) |
| Overshoot maximo (S) | 2.93% | 2.8631% | |
| Ko (ataque contra N) | 4.0451 | 4.0454 | ver nota de validacao abaixo |
| Overshoot real (ataque contra N) | 48.9% | 48.91% | |
| Overshoot maximo (ataque contra S) | 10.12% | 10.1368% | Ko=1.2815 (valor do artigo) |
| Fitness BSA medio (N) | 1.840e-07 | 2.031e-34 | n=30 (reduzido de 100) |
| Fitness BSA medio (S) | 7.420e-04 | 8.325e-04 | |

**Nota de validacao (Ko / overshoot contra N):** Ko projetado (4.0454) e overshoot real obtido (48.91%) reproduzem os valores de referencia (Ko=4.0451, overshoot=48.90%) com erro abaixo de 0.1%. Ver src/ncs_model.py (nota 'RESOLUCAO DE AMBIGUIDADE DE SINAL') para a convencao de sinal usada na planta e no controlador.

**Nota de modelagem (controlador comutante):** Estado do controlador inativo congelado (sem reset especial na retomada); confirmado diretamente contra o modelo Simulink original do autor (ver README, secao 'Comparacao com o codigo-fonte original').

**Nota sobre o IC95% do settling time:** O '0.0146s' reportado como referencia e um intervalo de confianca de 95% da MEDIA (conforme o texto da fonte: 'mean is 4.2827s +-0.0146s, with a confidence interval of 95%'), nao o desvio-padrao da distribuicao entre simulacoes -- um desvio-padrao de 0.0146s seria incompativel com a faixa min/max de 2.88-6.42s reportada na mesma fonte. Por isso comparamos o desvio-padrao replicado com o INTERVALO DE CONFIANCA DA MEDIA que ele implica (1.96*std/sqrt(n)), nao diretamente com o desvio-padrao bruto.

**Nota de escala (BSA):** Reduzido de 100 para 30 repeticoes por execucao (ver src/bsa.py); populacao=100 e iteracoes=600 mantidos identicos ao artigo (problema 2D, barato). Identificacao da planta (4D, 800 iter) NAO executada em escala completa -- documentada como extensao futura.

## Parte B - Parametros calibrados do jogo

| Parametro | Fonte artigo (baseline) | Fonte replicado (cross-check) |
|---|---|---|
| V | 1.0000 | 1.0000 |
| L | 0.9780 | 0.9782 |
| g | 0.2024 | 0.2027 |
| sigma | 0.7845 | 0.7848 |

Cenario-base (kappa, p, delta), aplicado aos dois: kappa=0.1, p=0.3, delta=1.0

kappa, p e delta nao tem contrapartida numerica direta na literatura de controle consultada; kappa=0.1, p=0.3, delta=1.0 (em unidades de V=1) sao um cenario-base ilustrativo. A analise de sensibilidade (Parte C) varre os tres em faixas amplas.

**Achado 'knife-edge':** Tanto a calibracao PRIMARIA (artigo, y*=1.0056) quanto a SECUNDARIA (replicada, y*=1.0063) colocam y* = sigma/(L(1-g)) ligeiramente ACIMA de 1: os numeros de controle (tanto os publicados quanto os que replicamos de forma independente) colocam o custo da comutacao e o beneficio de mitigacao quase exatamente em equilibrio ('knife-edge'). Nesse regime, o ponto interior do jogo deixa de existir dentro de (0,1)^2 e (N,C) passa a ser, tecnicamente, um equilibrio de Nash estrito (ver Tabela de vertices) -- ou seja, o proprio par controlador/contramedida descrito na literatura, com os parametros de deteccao/custo do cenario-base acima, esta no limite exato entre 'vale a pena comutar sempre' e 'a comutacao e uma estrategia mista'. Para ilustrar o comportamento GENERICO de centro/orbita fechada previsto pela teoria para o caso interior (o resultado central da Parte C), as Figs. C1-C2 usam um cenario PEDAGOGICO (nao calibrado): os mesmos V, L, g, kappa, p, delta do cenario-base, mas com sigma reduzido pela metade -- apenas para obter um ponto interior genuino a ilustrar.

## Parte C - Dinamica de replicador

- **Baseline (artigo):** x* = 0.8200, y* = 1.0056 (FORA de (0,1) -- ver achado knife-edge)
- **Replicado (cross-check):** x* = 0.8202, y* = 1.0063 (FORA de (0,1) -- mesmo achado knife-edge)
- **Pedagogico (sigma reduzido, nao calibrado):** x* = 0.8200, y* = 0.5028 (interior valido)
- Autovalores no ponto interior do cenario pedagogico (analitico): ['0.1777533333267264j', '-0.1777533333267264j']
- Classificado como centro (autovalores puramente imaginarios)? **True**
- Verificacao ergodica (Teorema 9.8 de Gintis, cenario pedagogico): media temporal (0.8199, 0.5025) vs (x*,y*) = (0.8200, 0.5028)
- Periodo estimado da orbita (pequena amplitude, cenario pedagogico): {'mean_period': 35.48435522006481, 'std_period': 1.7517238631424466e-09, 'n_cycles': 5}

### Equilibrios de Nash estritos nos vertices (cenario baseline/artigo)

| Vertice | (x,y) | Nash estrito? |
|---|---|---|
| (N,Ø) | (0,0) | False |
| (N,C) | (0,1) | True |
| (S,Ø) | (1,0) | False |
| (S,C) | (1,1) | False |

### Equilibrios de Nash estritos nos vertices (cenario pedagogico)

| Vertice | (x,y) | Nash estrito? |
|---|---|---|
| (N,Ø) | (0,0) | False |
| (N,C) | (0,1) | False |
| (S,Ø) | (1,0) | False |
| (S,C) | (1,1) | False |

### Regimes alternativos (sigma baixo/alto x p*delta baixo/alto, a partir do cenario baseline)

| Regime | x* | y* | Interior valido? | Nash estritos |
|---|---|---|---|---|
| sigma_baixo_pdelta_baixo | 1.003 | 0.300 | False | ['(S,C)'] |
| sigma_baixo_pdelta_alto | 0.237 | 0.300 | True | [] |
| sigma_alto_pdelta_baixo | 1.003 | 3.000 | False | ['(N,C)'] |
| sigma_alto_pdelta_alto | 0.237 | 3.000 | False | ['(N,C)'] |

## Figuras

| Figura | Arquivo | Descricao | Correspondencia no artigo |
|---|---|---|---|
| Fig. A1 | `figures/fig_A1_non_switching_response.png` | Resposta ao degrau, controlador nao comutante | artigo Fig. 5 (curva 'without attack') |
| Fig. A2a | `figures/fig_A2_switching_envelope.png` | Envoltoria de 100.000 simulacoes, controlador comutante | artigo Fig. 12 |
| Fig. A2b | `figures/fig_A2_settling_histogram.png` | Histograma do tempo de acomodacao, controlador comutante | artigo Fig. 13 |
| Fig. A3a | `figures/fig_A3_attack_non_switching.png` | Ataque SD-Controlled contra controlador nao comutante | artigo Fig. 5 |
| Fig. A3b | `figures/fig_A3_attack_switching.png` | Ataque SD-Controlled contra controlador comutante | artigo Fig. 14 |
| Fig. A4 | `figures/fig_A4_bsa_dispersion.png` | Dispersao dos coeficientes estimados por BSA (N vs S) | artigo Fig. 9 |
| Fig. C1 | `figures/fig_C1_phase_portrait.png` | Retrato de fase da dinamica de replicador | sem equivalente no artigo original (contribuicao deste trabalho) |
| Fig. C2 | `figures/fig_C2_time_series.png` | Series temporais x(t), y(t) | sem equivalente no artigo original |
| Fig. C3 | `figures/fig_C3_sensitivity.png` | Analise de sensibilidade de x*, y* | sem equivalente no artigo original |
