import numpy as np
import matplotlib.pyplot as plt
from scripts.caminhos import preparar_pasta_imagens
from scripts.modelos.lorenz import (
    calcular_pontos_equilibrio_lorenz,
    calcular_secao_poincare,
    projetar_pontos,
    projetar_ponto_equilibrio,
    runge_kutta_4,
    sistema_lorenz,
)

PASTA_IMAGENS = preparar_pasta_imagens()


# ============================================================
# Sistema de Lorenz
# ============================================================

# ============================================================
# Parâmetros da simulação
# ============================================================

sigma = 10
beta = 8 / 3

valores_rho = [10, 20, 28]
valores_perturbacao = [0, 10**-5]

tempo_inicial = 0
tempo_final = 50
passo = 0.01

tempo = np.arange(tempo_inicial, tempo_final, passo)


# Condição inicial usada como referência:
# x = 0.0
# y = 1.0 + e
# z = 20.0
def condicao_inicial(e):
    return np.array([0.0, 1.0 + e, 20.0])


# ============================================================
# Gráficos das histórias temporais
# ============================================================

figura, eixos = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(16, 10),
    sharex=True
)

variaveis = ["x(t)", "y(t)", "z(t)"]

for linha, rho in enumerate(valores_rho):
    for e in valores_perturbacao:
        estado_inicial = condicao_inicial(e)

        solucao = runge_kutta_4(
            sistema_lorenz,
            estado_inicial,
            tempo,
            sigma,
            beta,
            rho
        )

        if e == 0:
            legenda = "e = 0"
        else:
            legenda = r"e = $10^{-5}$"

        for coluna in range(3):
            eixos[linha, coluna].plot(
                tempo,
                solucao[:, coluna],
                linewidth=0.8,
                label=legenda
            )

    for coluna in range(3):
        eixos[linha, coluna].set_title(f"{variaveis[coluna]} para ρ = {rho}")
        eixos[linha, coluna].set_xlabel("Tempo")
        eixos[linha, coluna].set_ylabel(variaveis[coluna])
        eixos[linha, coluna].grid(True)
        eixos[linha, coluna].legend()

figura.suptitle(
    "História temporal do sistema de Lorenz com perturbação nas condições iniciais",
    fontsize=16
)

plt.tight_layout()
plt.savefig(PASTA_IMAGENS / "historia_temporal_lorenz_comparacao.png", dpi=300)
plt.show()


# ============================================================
# Gráficos no espaço de fase
# ============================================================

figura_3d = plt.figure(figsize=(16, 5))

for indice, rho in enumerate(valores_rho):
    eixo_3d = figura_3d.add_subplot(1, 3, indice + 1, projection="3d")

    for e in valores_perturbacao:
        estado_inicial = condicao_inicial(e)

        solucao = runge_kutta_4(
            sistema_lorenz,
            estado_inicial,
            tempo,
            sigma,
            beta,
            rho
        )

        if e == 0:
            legenda = "e = 0"
        else:
            legenda = r"e = $10^{-5}$"

        eixo_3d.plot(
            solucao[:, 0],
            solucao[:, 1],
            solucao[:, 2],
            linewidth=0.6,
            label=legenda
        )

    eixo_3d.set_title(f"Espaço de fase para ρ = {rho}")
    eixo_3d.set_xlabel("x")
    eixo_3d.set_ylabel("y")
    eixo_3d.set_zlabel("z")
    eixo_3d.legend()

figura_3d.suptitle(
    "Espaço de fase do sistema de Lorenz para diferentes valores de ρ",
    fontsize=16
)

plt.tight_layout()
plt.savefig(PASTA_IMAGENS / "espaco_fase_lorenz_comparacao.png", dpi=300)
plt.show()


# ============================================================
# Funções auxiliares para Seção de Poincaré
# ============================================================

def ponto_pertence_ao_corte(ponto, plano, valor_corte, tolerancia=0.20):
    """
    Verifica se o ponto de equilíbrio está próximo do corte analisado.
    """
    x, y, z = ponto

    if plano == "yz":
        return abs(x - valor_corte) <= tolerancia

    if plano == "xz":
        return abs(y - valor_corte) <= tolerancia

    if plano == "xy":
        return abs(z - valor_corte) <= tolerancia

    raise ValueError("Plano inválido. Use 'yz', 'xz' ou 'xy'.")


def ajustar_limites_dinamicos(eixo, xs, ys, plano):
    """
    Ajusta os limites dos eixos automaticamente com uma margem.
    """
    if len(xs) == 0 or len(ys) == 0:
        if plano == "yz":
            eixo.set_xlim(-9, 9)
            eixo.set_ylim(5, 30)
        elif plano == "xz":
            eixo.set_xlim(-9, 9)
            eixo.set_ylim(5, 30)
        elif plano == "xy":
            eixo.set_xlim(-9, 9)
            eixo.set_ylim(-9, 9)
        return

    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)

    largura_x = xmax - xmin
    largura_y = ymax - ymin

    margem_x = max(0.6, 0.12 * largura_x)
    margem_y = max(0.6, 0.12 * largura_y)

    if largura_x == 0:
        margem_x = 1.0

    if largura_y == 0:
        margem_y = 1.0

    eixo.set_xlim(xmin - margem_x, xmax + margem_x)
    eixo.set_ylim(ymin - margem_y, ymax + margem_y)


def aplicar_limites_do_artigo(eixo, plano, valor_corte):
    """
    Aplica limites fixos nos eixos para manter a mesma janela visual
    da figura de referência do artigo.

    Atenção aos nomes dos planos:
    - plano yz, em x = c  -> eixo horizontal: y | eixo vertical: z
    - plano xz, em y = c  -> eixo horizontal: x | eixo vertical: z
    - plano xy, em z = c  -> eixo horizontal: x | eixo vertical: y
    """
    limites = {
        # Coluna 1: planos yz, com x constante
        # Exemplo citado: plano yz, em x = -10, horizontal de -9 a 6 e vertical de 0 a 40.
        ("yz", -10): ((-10, 8),  (0, 45)),
        ("yz", -5):  ((-6, 8),  (0, 40)),
        ("yz", 0):   ((-1, 8), (0, 35)),
        ("yz", 5):   ((4, 11),  (0, 25)),
        ("yz", 10):  ((4, 20),  (7.5, 27.5)),

        # Coluna 2: planos xz, com y constante
        ("xz", -10): ((-20, 10),   (0, 50)),
        ("xz", -5):  ((-20, 10),   (0, 50)),
        ("xz", 0):   ((-20, 10),   (0, 50)),
        ("xz", 5):   ((-12.5, 10),   (0, 45)),
        ("xz", 10):  ((4, 9), (0, 25)),

        # Coluna 3: planos xy, com z constante
        ("xy", 5):   ((-6, 8),   (-15, 15)),
        ("xy", 10):  ((-10, 10), (-20, 20)),
        ("xy", 15):  ((-15, 15), (-30, 30)),
        ("xy", 20):  ((-15, 15), (-30, 30)),
        ("xy", 25):  ((-20, 20), (-30, 30)),
    }

    xlim, ylim = limites[(plano, valor_corte)]
    eixo.set_xlim(*xlim)
    eixo.set_ylim(*ylim)


# ============================================================
# Seções de Poincaré em múltiplos cortes
# ============================================================

valores_rho_poincare = [10, 15, 20, 28]
rhos_com_ponto_estabilizacao = [10, 15, 20]

tempo_poincare = np.arange(0, 200, passo)

# Não descarta transiente: começa a seção de Poincaré em t = 0,
# como no artigo usado para validação.
tempo_transiente = 0
indice_transiente = int(tempo_transiente / passo)

# Cortes usados na figura de referência do artigo.
cortes_yz = [-10, -5, 0, 5, 10]       # x constante, projeção yz
cortes_xz = [-10, -5, 0, 5, 10]       # y constante, projeção xz
cortes_xy = [5, 10, 15, 20, 25]       # z constante, projeção xy

cores_rho = {
    10: "green",
    15: "red",
    20: "blue",
    28: "black"
}

# tamanhos maiores para enxergar melhor rho = 10 e 15
tamanhos_rho = {
    10: 24,
    15: 22,
    20: 18,
    28: 14
}

cores_estabilizacao = {
    10: "green",
    15: "red",
    20: "blue"
}


# ============================================================
# Pré-calcula as soluções para cada rho
# ============================================================

solucoes_por_rho = {}

for rho in valores_rho_poincare:
    solucao = runge_kutta_4(
        sistema_lorenz,
        condicao_inicial(0),
        tempo_poincare,
        sigma,
        beta,
        rho
    )

    solucoes_por_rho[rho] = solucao[indice_transiente:]


# ============================================================
# Configuração dos 15 gráficos
# ============================================================

configuracoes_graficos = []

for corte in cortes_yz:
    configuracoes_graficos.append({
        "plano": "yz",
        "indice_variavel": 0,
        "valor_corte": corte,
        "titulo": f"Plano yz, em x = {corte}",
        "xlabel": "y",
        "ylabel": "z"
    })

for corte in cortes_xz:
    configuracoes_graficos.append({
        "plano": "xz",
        "indice_variavel": 1,
        "valor_corte": corte,
        "titulo": f"Plano xz, em y = {corte}",
        "xlabel": "x",
        "ylabel": "z"
    })

for corte in cortes_xy:
    configuracoes_graficos.append({
        "plano": "xy",
        "indice_variavel": 2,
        "valor_corte": corte,
        "titulo": f"Plano xy, em z = {corte}",
        "xlabel": "x",
        "ylabel": "y"
    })


# ============================================================
# Figura final com 15 gráficos
# ============================================================

figura_poincare, eixos = plt.subplots(
    nrows=5,
    ncols=3,
    figsize=(17, 22)
)

for linha in range(5):

    configs_da_linha = [
        configuracoes_graficos[linha],          # yz
        configuracoes_graficos[linha + 5],      # xz
        configuracoes_graficos[linha + 10]      # xy
    ]

    for coluna, config in enumerate(configs_da_linha):
        eixo = eixos[linha, coluna]

        plano = config["plano"]
        indice_variavel = config["indice_variavel"]
        valor_corte = config["valor_corte"]

        xs_totais = []
        ys_totais = []

        # Pontos de Poincaré para cada valor de rho
        for rho in valores_rho_poincare:
            solucao_regime = solucoes_por_rho[rho]

            pontos_secao = calcular_secao_poincare(
                solucao_regime,
                indice_variavel=indice_variavel,
                valor_plano=valor_corte,
                sentido="positivo"
            )

            eixo_x, eixo_y = projetar_pontos(
                pontos_secao,
                plano
            )

            if len(eixo_x) > 0:
                xs_totais.extend(eixo_x)
                ys_totais.extend(eixo_y)

            eixo.scatter(
                eixo_x,
                eixo_y,
                s=tamanhos_rho[rho],
                alpha=0.85,
                color=cores_rho[rho],
                label=rf"$\rho$ = {rho}"
            )

        # Pontos de estabilização para rho = 10, 15 e 20.
        # A figura de referência usa a projeção do ponto de equilíbrio positivo
        # como marcador em todos os cortes, sem exigir que ele pertença ao plano.
        for rho in rhos_com_ponto_estabilizacao:
            ponto_eq = calcular_pontos_equilibrio_lorenz(beta, rho)[0]
            x_eq, y_eq = projetar_ponto_equilibrio(
                ponto_eq,
                plano
            )

            xs_totais.append(x_eq)
            ys_totais.append(y_eq)

            eixo.scatter(
                x_eq,
                y_eq,
                s=100,
                marker="^",
                color=cores_estabilizacao[rho],
                edgecolors="black",
                linewidths=0.8,
                label=rf"Estab. $\rho$ = {rho}"
            )

        eixo.set_title(config["titulo"])
        eixo.set_xlabel(config["xlabel"])
        eixo.set_ylabel(config["ylabel"])
        eixo.grid(True)

        aplicar_limites_do_artigo(
            eixo,
            plano,
            valor_corte
        )


# ============================================================
# Legenda única
# ============================================================

handles = []
labels = []

for eixo_linha in eixos:
    for eixo in eixo_linha:
        h, l = eixo.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

legenda_unica = dict(zip(labels, handles))

figura_poincare.legend(
    legenda_unica.values(),
    legenda_unica.keys(),
    loc="lower center",
    ncol=4,
    fontsize=11
)

figura_poincare.suptitle(
    "Seções de Poincaré para o sistema de Lorenz em diferentes planos de corte",
    fontsize=18
)

plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig(PASTA_IMAGENS / "poincare_lorenz_15_cortes_zoom.png", dpi=300)
plt.show()