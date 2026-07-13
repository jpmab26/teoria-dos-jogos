# Comutação de controladores como defesa em sistemas de controle em rede: uma leitura via Teoria dos Jogos Evolutiva

Este repositório faz duas coisas:

1. **Replica computacionalmente** um sistema de controle em rede (Networked
   Control System, NCS) atacado por um adversário que primeiro *identifica*
   o controlador e a planta por escuta passiva da rede, e depois usa esse
   modelo para *injetar dados* de forma furtiva; e a contramedida proposta
   para esse ataque, um controlador que alterna aleatoriamente entre duas
   funções de controle para dificultar a identificação.
2. **Modela o conflito entre operador e atacante como um jogo evolutivo
   assimétrico de duas populações**, simulado via dinâmica de replicador,
   para responder a uma pergunta que a simulação de controle sozinha não
   responde: *dado que comutar tem um custo (settling time maior) e que
   atacar tem um custo (esforço computacional, risco de detecção), quando é
   racional, para cada lado, adotar sua estratégia mais cara?*

A parte 2 (Teoria dos Jogos) é o eixo central deste trabalho — a parte 1
existe para dar números realistas aos parâmetros do jogo.

## Como rodar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

Tempo de execução total: ~30-40s (a maior parte gasta nas 100.000 simulações
de Monte Carlo do controlador comutante e nas 2×30 execuções do BSA).

Saídas geradas:
- `figures/*.png` — todas as figuras (9 no total).
- `resultados.json` — todos os números, em formato estruturado.
- `resultados.md` — tabela de validação artigo × replicado, parâmetros
  calibrados, limiares (x\*, y\*), classificação de estabilidade e índice de
  figuras.

Todas as fontes de aleatoriedade usam `numpy.random.default_rng(seed)` com
seeds fixas, definidas em `SEEDS` no topo de `main.py`, de modo que rodar
`main.py` duas vezes produz exatamente os mesmos números.

## Estrutura do projeto

```
src/
  ncs_model.py    # planta + controladores em espaço de estados, simulação
                   # em malha fechada (não comutante e Monte Carlo do
                   # comutante), métricas de settling time/overshoot
  attack.py       # projeto do ganho de ataque Ko e simulação do ataque de
                   # injeção de dados contra o controlador único e o comutante
  bsa.py          # Backtracking Search Algorithm (identificação do
                   # controlador via escuta passiva), em escala reduzida
  calibration.py  # traduz os resultados de controle (Parte A) em payoffs
                   # do jogo evolutivo (V, L, g, sigma)
  replicator.py   # a matemática do jogo: limiares, dinâmica de replicador,
                   # Jacobiano, classificação de equilíbrios, ergodicidade
main.py           # orquestrador: roda tudo, gera figuras e relatórios
```

---

## Parte A — O sistema de controle em rede e o ataque

Um NCS é uma planta física controlada por um controlador digital através de
uma rede (Fig. conceitual: referência → controlador → rede → atuador →
planta → sensor → rede → de volta ao controlador). Este projeto usa como
caso de estudo um motor DC cuja velocidade rotacional é controlada por um
controlador Proporcional-Integral (PI), amostrado a 50 amostras/s:

```
G(z) = (g1·z + g2) / (z² + g3·z + g4)      # planta (motor DC)
C1(z) = (c1,1·z + c2,1) / (z - 1)          # controlador PI único
```

O **ataque** que este NCS sofre tem duas etapas:

1. **Identificação Passiva de Sistemas.** Um atacante que apenas escuta
   (eavesdropping) o tráfego de rede — sem precisar injetar nada — captura
   os sinais de entrada/saída do controlador e da planta durante uma janela
   de alguns segundos, e usa uma metaheurística bio-inspirada (Backtracking
   Search Algorithm, BSA) para ajustar um modelo estimado até que ele
   reproduza o comportamento observado. Isso dá ao atacante uma cópia
   aproximada de `G(z)` e `C(z)` sem qualquer acesso privilegiado.
2. **Injeção de dados controlada.** Com esse modelo em mãos, o atacante
   insere um ganho `M(z) = Ko` no caminho direto (entre o controlador e o
   atuador) — um ataque man-in-the-middle — dimensionado por análise de
   lugar das raízes para produzir um efeito físico específico e furtivo
   (por exemplo, um sobressinal de exatamente 50% na velocidade do motor):
   grande o suficiente para degradar o serviço, pequeno o suficiente para
   não ser obviamente percebido como um ataque.

A **contramedida** avaliada é um **controlador comutante aleatório**: em vez
de manter `C1(z)` fixo, o sistema alterna entre `C1(z)` e um segundo
controlador `C2(z)` (mais fraco, mas estável isoladamente), trocando em
instantes aleatórios sorteados de uma distribuição uniforme entre `a` e `b`
amostras (aqui, `a=40`, `b=60`, ou seja, entre 0,8s e 1,2s). A ideia é que,
sem saber quando as trocas ocorrem, o atacante não consegue mais ajustar um
único modelo `C(z)` que descreva o controlador com precisão — o modelo
estimado fica impreciso/ambíguo — e mesmo que ele descubra as duas funções
de controle, ainda não sabe qual delas está ativa em cada instante, o que
atrapalha o dimensionamento do ataque de injeção.

Do ponto de vista de controle, essa contramedida tem um **custo**: a
comutação aumenta o tempo de acomodação (settling time) da planta, porque a
cada troca o sinal de controle sofre uma pequena descontinuidade que a
planta precisa absorver.

### O que foi replicado e quão preciso ficou

Reconstruímos numericamente esse NCS inteiro — a planta, os dois
controladores, a regra de comutação markoviana, o ataque de injeção de
dados e (em escala reduzida) o próprio algoritmo BSA de identificação — e
comparamos contra os valores de referência da literatura de origem sobre
este NCS e sua contramedida. A tabela completa está em `resultados.md`;
resumo dos números principais:

| Grandeza | Referência | Replicado | Erro |
|---|---|---|---|
| Settling time, controlador único | 2,40 s | 2,4000 s | ~0% |
| Overshoot, controlador único | 0% | ~0% | — |
| Settling time médio, controlador comutante | 4,2827 s | 4,2835 s | ~0,02% |
| Settling time mín./máx., controlador comutante | 2,88 / 6,42 s | 2,88 / 6,46 s | ~0% / ~0,6% |
| Overshoot máximo, controlador comutante | ≤2,93% | 2,86% | — |
| Ganho de ataque Ko (contra controlador único, alvo 50%) | 4,0451 | 4,0454 | <0,1% |
| Overshoot real do ataque contra controlador único | 48,90% | 48,91% | <0,1% |
| Overshoot máximo do ataque contra controlador comutante | 10,12% | 10,14% | ~0,2% |

Esse nível de precisão não veio de ajuste fino de parâmetros — veio de
**resolver duas ambiguidades genuínas de implementação comparando
diretamente contra o código-fonte MATLAB/Simulink original do autor**, que
tivemos acesso a uma cópia de (pasta local `Ataque - Man-in-the-middle
(JISA)/`, não versionada neste repositório por não ser código nosso — ver
`.gitignore`). Duas descobertas concretas:

**1. Ambiguidade de sinal na fórmula da planta e do controlador, resolvida.**
A notação usual para esse tipo de planta (`G(z) = (g1·z+g2)/(z² − g3·z +
g4)`, com `g3` sendo um valor negativo) é ambígua quanto a se o segundo
coeficiente do denominador deveria entrar com o sinal literal ou invertido
— o mesmo vale para o segundo coeficiente do numerador do controlador PI.
Antes de ter acesso ao código original, testamos exaustivamente as 8
combinações de sinal possíveis (denominador da planta × numerador do
controlador) contra três critérios indiretos (estabilidade em malha
fechada, plausibilidade física dos polos, forma qualitativa da resposta ao
ataque) e chegamos a uma combinação candidata. O script MATLAB original
(`avalia_varios_chaveamentos.m`) **confirma exatamente essa escolha**:

```matlab
planta_num = [.3379 .2793];
planta_den = [1 -1.5462 .5646];   % MATLAB tf(): representa z² − 1.5462·z + 0.5646
num_cont   = [.1701 -.1673];      % C1(z) = (0.1701·z − 0.1673) / (z − 1)
den_cont   = [1 -1];
num_cont2  = 0.001*[1 0.2];       % C2(z) = (0.001·z + 0.0002) / (z − 1)
den_cont2  = [1 -1];
```

Como `g3 = -1,5462` é o valor publicado, `planta_den = [1, -1.5462, 0.5646]`
é exatamente `z² + g3·z + g4` — ou seja, o coeficiente entra **sem
inversão de sinal** —, e `num_cont = [0.1701, -0.1673] = [c1,1, c2,1]`
igualmente **sem inversão**. É exatamente a convenção adotada em
`plant_block`/`controller1_block`, em `src/ncs_model.py`, e é o que produz
o ajuste fino tabelado acima (a leitura literal ingênua da fórmula, com o
sinal invertido, embora também estável, erra o settling time em ~50% e o
Ko de ataque em ~500%).

**2. Mecanismo de comutação, confirmado.** O script cria os dois
controladores como blocos `DiscreteTransferFcn` independentes no Simulink
(`Controlador2`, `Controlador3`) e usa dois blocos `Switch` (critério
padrão `u2 ≥ 0`, alimentados pelo mesmo sinal de chaveamento aleatório
±1) para rotear o sinal de erro: **o controlador inativo recebe entrada
zero**, não o erro real — não há nenhuma transferência "bumpless" (sem
solavanco) explícita. Isso não é o mesmo que "congelar" o estado
literalmente, mas é **matematicamente equivalente** para os controladores
deste sistema: como ambos têm a forma de um integrador puro
(`x(k+1) = x(k) + 1·e(k)`), alimentar `e(k)=0` enquanto inativo dá
`x(k+1) = x(k)` — o estado fica constante de qualquer forma. Confirma
exatamente a opção `bumpless=False` (estado do controlador inativo
congelado) já adotada como padrão em `simulate_switching_monte_carlo`, em
`src/ncs_model.py` — não por ser a que melhor ajustava os números
(embora seja), mas por ser, de fato, o que o código original faz.

**3. Critério de tempo de acomodação, corrigido.** O script usa
`abs(1-respostas) > exp(-4)` para determinar se uma amostra ainda está fora
da faixa de acomodação — uma tolerância de `exp(-4) ≈ 1,83%`, não o
critério usual de 2% ou 5% de engenharia de controle. Ajustamos
`settling_time` (`src/ncs_model.py`) para usar exatamente essa tolerância
por padrão (constante `SETTLING_TOL`), o que sozinho levou o settling time
do controlador único de 2,28s (erro de ~5%) para 2,4000s (erro ~0%).

**4. O "±0,0146s" publicado é um intervalo de confiança da média, não um
desvio-padrão.** A distribuição do tempo de acomodação sob o controlador
comutante tem uma faixa mín-máx de 2,88s a 6,42s — incompatível com um
desvio-padrão de apenas 0,0146s (isso exigiria uma distância de centenas de
desvios-padrão entre o mínimo/máximo e a média, estatisticamente
impossível em 100.000 amostras). O texto de origem, lido com atenção,
confirma: é "a média... com um intervalo de confiança de 95%", não o
desvio-padrão bruto. Nosso desvio-padrão replicado (0,73s) implica um IC95%
da média de `1,96·0,73/√100000 ≈ 0,0045s` — mesma ordem de grandeza,
mesma conclusão qualitativa (a média está muito bem determinada apesar da
distribuição individual ser larga) —, mas comparável ao valor publicado
apenas como IC da média, não como desvio-padrão da distribuição. Ver a
nota "Nota sobre o IC95% do settling time" em `resultados.md`.

A identificação via BSA (Parte A.4) roda em escala reduzida (30 em vez de
100 repetições, por economia de tempo; ver `src/bsa.py`) mas reproduz
claramente o fenômeno central: sob o controlador único, o BSA identifica os
coeficientes corretos com precisão numérica exata; sob o controlador
comutante, as estimativas ficam dispersas e não convergem para nenhum dos
dois controladores reais — a "assinatura" que o jogo da Parte B/C usa para
calibrar o parâmetro `g` (razão de mitigação). Não tivemos acesso a um
script MATLAB equivalente para a identificação via BSA propriamente dita
(o script `ataque_mitm_perda_0.m` na pasta original usa `sisotool`, a
ferramenta *interativa* de projeto do MATLAB, para desenhar `M(z)` a partir
de coeficientes já identificados — não reproduz o processo de otimização do
BSA em si), então essa parte permanece validada apenas indiretamente (pelo
padrão de dispersão N vs. S, não por comparação numérica direta de código).

---

## Parte B/C — O jogo evolutivo: motivação, modelo e simulação

### Por que modelar isso como um jogo

A Parte A responde "o controlador comutante funciona, do ponto de vista de
engenharia de controle?" (sim: mitiga o ataque, ao custo de um settling
time maior). Mas não responde a uma pergunta igualmente importante, que é
estratégica, não apenas técnica: **um operador de NCS enfrenta uma
população de atacantes com comportamento heterogêneo e desconhecido a
priori** — alguns nunca vão atacar (não têm acesso à rede, ou avaliam que
não compensa o risco), outros vão atacar sempre que acharem lucrativo. Do
mesmo modo, o operador não decide de uma vez por todas se usa o
controlador comutante — diferentes operadores, ou o mesmo operador ao
longo do tempo, ajustam essa escolha conforme observam o que funciona.
Essa dinâmica de "cada lado ajusta sua estratégia com base no que tem dado
certo para os outros" é exatamente o que a Teoria dos Jogos Evolutiva (EGT)
foi desenhada para capturar — ao contrário da teoria dos jogos clássica,
que assume jogadores hiper-racionais escolhendo uma única jogada ótima de
uma vez, a EGT modela **populações** de jogadores que herdam, imitam ou
aprendem estratégias, e cuja composição populacional evolui ao longo do
tempo em função do desempenho (payoff) relativo de cada estratégia.

### Jogadores, ações e a bimatriz de payoffs

O jogo tem **duas populações**, cada uma com **duas estratégias puras**.
Como o jogador da população D nunca joga contra outro jogador da população
D (o defensor sempre enfrenta um atacante, nunca outro defensor), este é um
**jogo assimétrico** — distinção que importa bastante para a análise de
equilíbrio, como veremos adiante.

- **População D (defensores/operadores do NCS).** `N` = manter o
  controlador único (menor custo operacional, settling time nominal); `S`
  = adotar o controlador comutante (settling time maior, mas dificulta a
  identificação do atacante).
- **População A (atacantes man-in-the-middle).** `C` = lançar o ataque
  completo (Identificação Passiva + Injeção de Dados); `Ø` = abster-se
  (por falta de acesso ao laço de controle, ou por avaliar que o ataque
  não compensa o custo/risco).

Cada célula da bimatriz abaixo é o par de payoffs `(π_D, π_A)` resultante
do encontro entre uma estratégia de defesa (linha) e uma estratégia de
ataque (coluna):

| | `C` (atacar) | `Ø` (abster-se) |
|---|---|---|
| **`N`** | `−L`, `V−κ` | `0`, `0` |
| **`S`** | `−σ−gL`, `gV−κ−pδ` | `−σ`, `0` |

Os sete parâmetros, todos não-negativos:

- **`V`** — valor, para o atacante, de uma degradação furtiva
  bem-sucedida (normalizamos `V=1`; corresponde ao sobressinal-alvo de 50%
  que o atacante pretende induzir).
- **`L`** — perda do defensor quando o ataque atinge um controlador único
  sem sucesso de mitigação (calibrado a partir do sobressinal real
  obtido contra `N`, ~49% — ou seja, o ataque atinge quase exatamente o
  que pretendia, `L≈V`).
- **`g ∈ [0,1]`** — **razão de mitigação**: fração do efeito pretendido
  pelo atacante que de fato se concretiza quando o defensor usa `S`
  (calibrado a partir do overshoot real observado contra o controlador
  comutante, ~10%, dividido pelo alvo de 50% ⇒ `g≈0,20`).
- **`σ`** — custo operacional da comutação para o defensor (calibrado a
  partir do aumento *relativo* do settling time ao adotar `S`,
  `(4,28−2,4)/2,4 ≈ 0,78`).
- **`κ`** — custo do ataque para o atacante (esforço computacional do BSA
  e da infraestrutura de interceptação; sem contrapartida numérica direta
  na literatura de controle consultada — tratado como parâmetro de
  projeto varrido em análise de sensibilidade).
- **`p`** — probabilidade de detecção do atacante quando o ataque falha
  sob `S` (o comportamento imprevisível da planta sob mitigação pode
  chamar a atenção de um observador); também tratado como parâmetro de
  projeto.
- **`δ`** — penalidade ao atacante quando detectado (perda de
  furtividade/atribuição); idem.

Lendo cada célula: sob `(N,Ø)` ninguém ganha nem perde nada. Sob `(N,C)`, o
ataque é executado contra um controlador único e tem sucesso pleno: o
defensor sofre a perda integral `−L`, o atacante obtém o valor líquido do
ataque, `V−κ`. Sob `(S,Ø)`, o defensor paga o custo da comutação mesmo sem
que haja ataque para mitigar (`−σ`), e o atacante, abstendo-se, não ganha
nem perde. Sob `(S,C)`, o ataque é lançado contra um controlador comutante:
o defensor sofre uma perda mitigada (`−σ − gL`, o custo da comutação somado
à fração `g` da perda que ainda se concretiza), e o atacante obtém um valor
proporcionalmente mitigado (`gV`), descontado do seu custo `κ` e de uma
penalidade esperada de detecção `pδ`.

### Por que dinâmica de replicador, e como ela é derivada

Seja `x ∈ [0,1]` a fração da população D jogando `S`, e `y ∈ [0,1]` a
fração da população A jogando `C`. A ideia central da dinâmica de
replicador é que **a fração de uma população jogando uma estratégia cresce
proporcionalmente a quanto essa estratégia está performando acima da
média da população** — em outras palavras, estratégias com payoff acima da
média ganham espaço; estratégias abaixo da média perdem espaço.

Para a população D, dado que a população A joga `C` com frequência `y`, o
payoff esperado de cada ação do defensor é a média ponderada das colunas da
bimatriz:

```
π_D(N | y) = y·(−L) + (1−y)·0        = −yL
π_D(S | y) = y·(−σ−gL) + (1−y)·(−σ)  = −σ − ygL
```

A diferença `π_D(S|y) − π_D(N|y) = yL(1−g) − σ` diz exatamente quanto `S`
está performando acima (se positivo) ou abaixo (se negativo) de `N`; essa
diferença, multiplicada por `x(1−x)` (que garante que a fração pare de
crescer nos extremos `x=0` e `x=1`, e que zera a força evolutiva quando a
população já é homogênea), é a equação de replicador para `x`:

```
ẋ = x(1−x) · [ y·L·(1−g) − σ ]
```

De forma simétrica para a população A (dado que a população D joga `S` com
frequência `x`):

```
π_A(C | x) = x·(gV−κ−pδ) + (1−x)·(V−κ) = (V−κ) − x·[(1−g)V + pδ]
π_A(Ø | x) = 0

ẏ = y(1−y) · [ (V−κ) − x·((1−g)V + pδ) ]
```

Essas duas equações — implementadas em `replicator_rhs`, em
`src/replicator.py` — são o motor de toda a Parte C.

### Os limiares x\* e y\*: o ponto onde a melhor resposta se inverte

Cada colchete acima muda de sinal em um ponto específico. Chamamos esses
pontos de `y*` (o limiar de ataques acima do qual vale a pena o defensor
comutar) e `x*` (o limiar de defensores comutando acima do qual o ataque
deixa de compensar):

```
y* = σ / (L·(1−g))                      #  ẋ > 0  ⟺  y > y*
x* = (V−κ) / ((1−g)·V + p·δ)             #  ẏ > 0  ⟺  x < x*
```

Interpretação de `y*`: se a fração observada de atacantes ultrapassa esse
limiar, o custo da comutação (`σ`) fica compensado pela perda evitada
(`yL(1−g)`), e a população de defensores migra para `S`; abaixo do limiar,
migra de volta para `N`. Simetricamente, `x*` é a fração de defensores
comutando acima da qual o ataque mitigado (`gV`) já não cobre seu custo e
risco (`κ+pδ`), e a população de atacantes migra para `Ø`.

### Equilíbrios de Nash nos quatro vértices

Os quatro cantos do quadrado `[0,1]²` — `(N,Ø)`, `(N,C)`, `(S,Ø)`, `(S,C)`
— são candidatos naturais a equilíbrio de Nash **estrito** (nenhum dos dois
jogadores ganha ao desviar unilateralmente). A condição de cada vértice
sai diretamente do sinal dos colchetes acima, avaliado em `y∈{0,1}` e
`x∈{0,1}`:

| Vértice | Condição para ser Nash estrito |
|---|---|
| `(N,Ø)` | Nunca, desde que `V > κ` (i.e., o ataque pleno contra `N` seja lucrativo — a própria razão de existir da contramedida) |
| `(N,C)` | `σ ≥ L(1−g)`, i.e. `y* ≥ 1` (e `V>κ`) |
| `(S,Ø)` | Nunca, pois `σ > 0` (não há razão para pagar o custo da comutação sem nenhum ataque a mitigar) |
| `(S,C)` | `σ < L(1−g)` **e** `gV ≥ κ+pδ`, i.e. `y*<1` **e** `x*≥1` |

Sob os parâmetros calibrados (Parte B), tipicamente `V>κ`, `σ<L(1−g)` e
`gV<κ+pδ` — regime em que **nenhum** dos quatro vértices é equilíbrio de
Nash estrito, e a melhor resposta de cada população persegue um ciclo:
partindo de `(N,Ø)`, o atacante desvia para `C` (já que `V>κ`); o defensor
então desvia para `S`; o atacante, já mitigado por `g`, desvia de volta
para `Ø`; e o defensor, sem mais ataques a mitigar, desvia de volta para
`N`, fechando o ciclo. Essa estrutura de melhor-resposta cíclica é
estruturalmente idêntica à do clássico jogo da Moeda Combinada (Matching
Pennies), cujo único candidato remanescente a equilíbrio é o ponto interior
misto `(x*, y*)`.

### Por que o ponto interior não pode ser uma ESS — e por que isso importa

Aqui entra uma sutileza específica de **jogos assimétricos** em Teoria dos
Jogos Evolutiva. O conceito central de equilíbrio da EGT é a Estratégia
Evolutivamente Estável (ESS, Maynard Smith & Price): uma estratégia é ESS
se nenhuma mutante, surgindo em fração pequena, consegue um payoff maior
que a estratégia residente contra a própria população residente. Isso
pressupõe que "um mutante encontra seu próprio tipo" — o que faz sentido
em um jogo **simétrico**, onde os dois jogadores são intercambiáveis, mas
não em um jogo assimétrico como o nosso, em que um jogador é sempre
defensor e o outro sempre atacante (um defensor nunca "encontra" outro
defensor no jogo).

A solução padrão na literatura (Selten, 1980) é definir a "versão
simétrica" do jogo assimétrico, sorteando aleatoriamente qual população faz
o papel de jogador 1 a cada rodada. Um resultado central dessa literatura é
que uma ESS dessa versão simetrizada precisa necessariamente ser um
equilíbrio de Nash **estrito** — isto é, cada população precisa estar
polarizada em uma única estratégia pura —, de modo que **equilíbrios
mistos no interior nunca são evolutivamente estáveis em jogos
assimétricos** de duas populações. A versão dinâmica desse resultado
(Gintis, *Game Theory Evolving*, 1999, cap. 9) mostra que, sob a dinâmica
de replicador de duas populações, um equilíbrio de Nash em estratégia
mista interior nunca é um ponto assintoticamente estável; para o caso
específico de duas estratégias por população (nossa estrutura exata), o
ponto misto interior é sempre um ponto de sela instável **ou** um "ponto
focal evolutivo" — um centro, em torno do qual as trajetórias formam
órbitas fechadas, com a dinâmica sendo **ergódica** (a média temporal das
frequências ao longo de uma órbita coincide com as frequências de
equilíbrio).

Ou seja: quando nenhum vértice é Nash estrito (o regime calibrado deste
jogo), a teoria prevê que o único candidato a equilíbrio, o ponto interior
`(x*,y*)`, não vai ser um estado estável para onde o sistema converge — vai
ser, na melhor das hipóteses, um centro cíclico. Isso tem uma leitura
prática direta: **não existe uma fração "certa" e estável de defensores
comutando e atacantes atacando; o sistema oscila perpetuamente**, com
defensores e atacantes reagindo em ciclo às escolhas uns dos outros, nunca
se acomodando em um equilíbrio fixo.

### Confirmando isso analiticamente: o Jacobiano é sempre anti-simétrico

Este projeto não apenas confia na literatura citada acima — verificamos
diretamente, por conta própria, que o ponto interior deste jogo específico
é sempre um centro, linearizando o sistema em `(x*,y*)`.

Escrevendo `ẋ = x(1−x)·f(y)` e `ẏ = y(1−y)·h(x)`, com `f(y)=yL(1−g)−σ` e
`h(x)=(V−κ)−x[(1−g)V+pδ]`, o Jacobiano em `(x*,y*)` é:

```
∂ẋ/∂x = (1−2x)·f(y) + x(1−x)·f'(y)·0   →  em (x*,y*): (1−2x*)·f(y*) = (1−2x*)·0 = 0
∂ẋ/∂y = x(1−x)·f'(y) = x(1−x)·L(1−g)   →  em (x*,y*): x*(1−x*)·L(1−g)  =: a₁₂
∂ẏ/∂x = y(1−y)·h'(x) = −y(1−y)·[(1−g)V+pδ]  →  em (x*,y*): −y*(1−y*)·[(1−g)V+pδ]  =: a₂₁
∂ẏ/∂y = (1−2y)·h(x) + y(1−y)·0          →  em (x*,y*): (1−2y*)·h(x*) = (1−2y*)·0 = 0
```

A diagonal principal se anula **identicamente** — não por coincidência
numérica, mas porque `f(y*)=0` e `h(x*)=0` são exatamente a definição dos
limiares. Sobra a matriz anti-simétrica `J = [[0, a₁₂], [a₂₁, 0]]`, cujos
autovalores resolvem `λ² = a₁₂·a₂₁`. Como `a₁₂>0` (produto de termos
positivos) e `a₂₁<0` (mesmo produto, com sinal trocado pelo `−`),
`a₁₂·a₂₁ < 0`, logo `λ² < 0` e `λ = ±i·√(−a₁₂·a₂₁)` é **sempre puramente
imaginário**, para qualquer valor positivo dos parâmetros do jogo — ou
seja, o ponto interior deste jogo 2×2 é **sempre** um centro, nunca um foco
ou uma sela, confirmando analiticamente (não apenas citando) o resultado
de Gintis. Essa derivação está implementada tanto analiticamente
(`jacobian_interior_analytic`) quanto numericamente por diferenças finitas
sobre a própria EDO (`jacobian_interior_numeric`, que não usa a fórmula
fechada — serve de checagem independente); os dois batem a 10+ casas
decimais em todos os cenários testados (ver `resultados.md`).

### Como simulamos tudo isso (metodologia computacional)

Todo o código relevante está em `src/replicator.py`; o que segue é o que
`main.py` executa, na ordem:

1. **Cálculo direto dos limiares** `x*(params)`, `y*(params)` — fórmulas
   fechadas, computadas para múltiplos cenários de parâmetros.
2. **Classificação dos 4 vértices** (`classify_vertices`) segundo as
   condições da tabela acima, para verificar quais (se algum) são
   equilíbrios de Nash estritos em cada cenário.
3. **Jacobiano no ponto interior**, calculado de duas formas independentes
   (fórmula analítica fechada `jacobian_interior_analytic`, e diferenças
   finitas centradas sobre a EDO `jacobian_interior_numeric`) — os
   autovalores de ambas são comparados para confirmar a classificação de
   centro.
4. **Integração numérica de trajetórias** (`simulate_trajectories`, via
   `scipy.integrate.solve_ivp`, método RK45 com tolerâncias `rtol=1e-9`,
   `atol=1e-11`) a partir de uma grade de 25 condições iniciais em
   `(0,1)²`, para desenhar o retrato de fase (Fig. C1) e confirmar
   visualmente as órbitas fechadas em torno de `(x*,y*)`.
5. **Verificação da propriedade ergódica** (`time_average_ergodicity_check`):
   integra uma única trajetória por um horizonte longo (2000 unidades de
   tempo) e compara a média temporal de `x(t)`, `y(t)` ao longo da órbita
   com `(x*,y*)` — se a dinâmica é de fato um centro ergódico, as duas
   devem coincidir, o que confirmamos numericamente.
6. **Estimativa do período da órbita** (`estimate_orbit_period`), via
   contagem de cruzamentos ascendentes de `y(t)` pelo limiar `y*` — como
   o centro é não-linear (ao contrário de um oscilador harmônico simples),
   o período depende da amplitude da órbita; reportamos o período de
   pequena amplitude (próximo ao ponto fixo), onde a aproximação linear é
   mais precisa.
7. **Análise de sensibilidade** (`sensitivity_sweep` e varreduras diretas
   em `main.py`): `x*` e `y*` recalculados enquanto se varia cada um dos
   parâmetros `κ`, `p·δ`, `σ/L` em faixas amplas (Fig. C3), e quatro
   regimes extremos (`σ` baixo/alto × `p·δ` baixo/alto) são checados
   quanto à existência de um ponto interior válido e de vértices Nash
   estritos.

### Da simulação de controle para os payoffs (Parte B)

`src/calibration.py` traduz os números de controle da Parte A nos quatro
parâmetros do jogo que têm contrapartida física direta:

```
V = 1                                          (normalização)
L = overshoot_real_contra_N / overshoot_alvo    (quão perto do pretendido o ataque chega)
g = overshoot_real_contra_S / overshoot_alvo    (razão de mitigação)
σ = (settling_time_S − settling_time_N) / settling_time_N   (custo relativo da comutação)
```

Os três parâmetros restantes (`κ`, `p`, `δ`) não têm contrapartida numérica
direta nos dados de controle disponíveis — são parâmetros de projeto,
fixados em um cenário-base ilustrativo (`κ=0,1`; `p=0,3`; `δ=1,0`, em
unidades de `V=1`) e variados amplamente na análise de sensibilidade
(item 7 acima).

### A Solução de Canto (Colapso do Equilíbrio Interior)

Com os parâmetros calibrados — tanto a partir da referência de controle
quanto a partir da nossa própria replicação, que agora concordam muito de
perto (dentro de ~0,1%) — `y* = σ/(L(1−g))` fica **ligeiramente acima de 1**
(≈1,006 a partir da referência; ≈1,006 a partir da nossa replicação). Como
`y*` é uma fração
(deveria estar entre 0 e 1 para representar um ponto interior válido),
`y*>1` significa que, **mesmo se todos os atacantes atacassem sempre**
(`y=1`), o benefício de mitigação ainda não compensaria o custo da
comutação o suficiente para justificá-la sob os parâmetros de detecção do
cenário-base — o ponto interior deixa de existir dentro de `(0,1)²`, e o
vértice `(N,C)` passa a ser, tecnicamente, um equilíbrio de Nash estrito
nesse cenário específico (ver tabela de vértices em `resultados.md`).

Interpretamos isso como um achado genuíno, não um artefato: os números de
controle desta contramedida específica colocam o custo de comutar e o
benefício de mitigação **quase exatamente em equilíbrio**. Pequenas
variações nos parâmetros de detecção (`κ`, `p`, `δ`) — que não têm uma
calibração empírica direta — bastam para empurrar o sistema para um regime
ou outro; a Fig. C3 (análise de sensibilidade) mostra exatamente essa
fronteira.

Para ilustrar o comportamento **genérico** de centro/órbita fechada
previsto pela teoria para o caso em que o ponto interior existe — o
resultado central e mais interessante da Parte C —, as Figs. C1 e C2 usam
um **cenário pedagógico, explicitamente não calibrado**: os mesmos `V, L,
g, κ, p, δ` do cenário-base, mas com `σ` artificialmente reduzido pela
metade, apenas o suficiente para trazer `y*` para dentro de `(0,1)`. Esse
cenário não representa uma segunda medição empírica — é uma ferramenta
didática para visualizar a órbita fechada, o Jacobiano anti-simétrico e a
propriedade ergódica descritos acima.

---

## Limitações e trabalhos futuros

- A identificação via BSA (Parte A.4) roda em escala reduzida (30
  repetições em vez de 100) e apenas para o controlador (problema 2D); a
  identificação da planta (problema 4D) não foi executada em escala
  completa — extensão direta se necessário.
- `κ`, `p`, `δ` não têm calibração empírica própria; a análise de
  sensibilidade (Fig. C3) cobre isso parcialmente, mas um trabalho futuro
  natural é estimar `p` diretamente da fração de trajetórias "extremas" nas
  simulações de Monte Carlo da Parte A (sinalizado como possibilidade em
  `calibration.py`).
- Extensões diretas do modelo canônico do jogo, não implementadas aqui:
  (a) uma terceira ação do atacante, um ataque não dependente de modelo
  atuando no laço de realimentação (que a contramedida de comutação não
  mitiga), tornando o jogo da população de atacantes um jogo de três
  estratégias puras; (b) uma versão de população finita com mutação, para
  investigar estabilidade estocástica e se o caráter aleatório da própria
  regra de comutação favorece a seleção de um dos vértices como estado
  estocasticamente estável sob perturbações persistentes.

## Referências

- de Sá, A. O.; da Costa Carmo, L. F. R.; Machado, R. C. S. *A controller
  design for mitigation of passive system identification attacks in
  networked control systems.* Journal of Internet Services and
  Applications, v. 9, n. 2, 2018.
- de Sá, A. O.; da Costa Carmo, L. F. R.; Machado, R. C. S. *Covert attacks
  in cyber-physical control systems.* IEEE Transactions on Industrial
  Informatics, v. 13, n. 4, p. 1641-1651, 2017.
- Civicioglu, P. *Backtracking search optimization algorithm for numerical
  optimization problems.* Applied Mathematics and Computation, v. 219, n.
  15, p. 8121-8144, 2013.
- Gintis, H. *Game Theory Evolving.* Princeton University Press, 1999
  (cap. 7: ESS; cap. 9: jogos evolutivos assimétricos e dinâmica de
  replicador de duas populações).
- Selten, R. *A note on evolutionarily stable strategies in asymmetric
  animal conflicts.* Journal of Theoretical Biology, v. 84, n. 1, p.
  93-101, 1980.
