
import numpy as np
import matplotlib.pyplot as plt
from scripts.caminhos import preparar_pasta_imagens
from scripts.modelos.lorenz import (
    calcular_pontos_equilibrio_lorenz,
    calcular_secao_poincare,
    projetar_pontos,
    runge_kutta_4,
    sistema_lorenz,
)

# ============================================================
# Configuração de saída
# ============================================================

PASTA_SAIDA = preparar_pasta_imagens()


def calcular_ponto_equilibrio_positivo(beta, rho):
    return calcular_pontos_equilibrio_lorenz(beta, rho)[0]


# ============================================================
# Parâmetros do caso de caos transiente
# ============================================================

sigma = 10
beta = 8 / 3
rho = 20

passo = 0.01
tempo = np.arange(0, 60, passo)

estado_inicial = np.array([0.0, 1.0, 20.0])

solucao = runge_kutta_4(
    sistema_lorenz,
    estado_inicial,
    tempo,
    sigma,
    beta,
    rho
)

ponto_eq = calcular_ponto_equilibrio_positivo(beta, rho)

# ============================================================
# Cortes e cores
# ============================================================

cortes_yz = [-10, -5, 0, 5, 10]
cores_yz = {
    -10: "tab:blue",
    -5: "tab:orange",
    0: "tab:green",
    5: "tab:red",
    10: "tab:purple",
}

cortes_xz = [-15, -10, -5, 0, 5, 10, 15]
cores_xz = {
    -15: "tab:blue",
    -10: "tab:orange",
    -5: "tab:green",
    0: "tab:red",
    5: "tab:purple",
    10: "tab:brown",
    15: "tab:pink",
}

cortes_xy = [5, 10, 15, 20, 25, 30]
cores_xy = {
    5: "tab:blue",
    10: "tab:orange",
    15: "tab:green",
    20: "tab:red",
    25: "tab:purple",
    30: "tab:brown",
}

# ============================================================
# Figura 28 - plano yz
# composição: bilateral em cinza + unilateral colorida + equilíbrio
# ============================================================

fig_28, ax_28 = plt.subplots(figsize=(10, 6))

for corte in cortes_yz:
    pontos_bi = calcular_secao_poincare(solucao, indice_variavel=0, valor_plano=corte, sentido="ambos")
    y_bi, z_bi = projetar_pontos(pontos_bi, "yz")
    ax_28.scatter(y_bi, z_bi, s=18, color="gray", alpha=0.75, label=None)

for corte in cortes_yz:
    pontos_uni = calcular_secao_poincare(solucao, indice_variavel=0, valor_plano=corte, sentido="positivo")
    y_uni, z_uni = projetar_pontos(pontos_uni, "yz")
    ax_28.scatter(y_uni, z_uni, s=22, color=cores_yz[corte], alpha=0.95, label=f"x= {corte}")

ax_28.scatter(
    ponto_eq[1], ponto_eq[2],
    s=90, marker="^", color="yellowgreen", edgecolors="black", linewidths=1.0,
    label="Última Leitura"
)

ax_28.set_title("Seção de Poincaré: plano yz.")
ax_28.set_xlabel("y")
ax_28.set_ylabel("z")
ax_28.set_xlim(-20, 20)
ax_28.set_ylim(4, 31)
ax_28.grid(False)
ax_28.legend(loc="upper right")
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "figura_28_poincare_caos_transiente_yz.png", dpi=300)

# ============================================================
# Figura 29 - plano xz
# ============================================================

fig_29, ax_29 = plt.subplots(figsize=(10, 6))

for corte in cortes_xz:
    pontos_bi = calcular_secao_poincare(solucao, indice_variavel=1, valor_plano=corte, sentido="ambos")
    x_bi, z_bi = projetar_pontos(pontos_bi, "xz")
    ax_29.scatter(x_bi, z_bi, s=18, color="gray", alpha=0.75, label=None)

for corte in cortes_xz:
    pontos_uni = calcular_secao_poincare(solucao, indice_variavel=1, valor_plano=corte, sentido="positivo")
    x_uni, z_uni = projetar_pontos(pontos_uni, "xz")
    ax_29.scatter(x_uni, z_uni, s=22, color=cores_xz[corte], alpha=0.95, label=f"y= {corte}")

ax_29.scatter(
    ponto_eq[0], ponto_eq[2],
    s=90, marker="^", color="yellowgreen", edgecolors="black", linewidths=1.0,
    label="Última Leitura"
)

ax_29.set_title("Seção de Poincaré: plano xz.")
ax_29.set_xlabel("x")
ax_29.set_ylabel("z")
ax_29.set_xlim(-20, 20)
ax_29.set_ylim(4, 31)
ax_29.grid(False)
ax_29.legend(loc="upper right")
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "figura_29_poincare_caos_transiente_xz.png", dpi=300)

# ============================================================
# Figura 30 - plano xy
# ============================================================

fig_30, ax_30 = plt.subplots(figsize=(10, 6))

for corte in cortes_xy:
    pontos_bi = calcular_secao_poincare(solucao, indice_variavel=2, valor_plano=corte, sentido="ambos")
    x_bi, y_bi = projetar_pontos(pontos_bi, "xy")
    ax_30.scatter(x_bi, y_bi, s=18, color="gray", alpha=0.75, label=None)

for corte in cortes_xy:
    pontos_uni = calcular_secao_poincare(solucao, indice_variavel=2, valor_plano=corte, sentido="positivo")
    x_uni, y_uni = projetar_pontos(pontos_uni, "xy")
    ax_30.scatter(x_uni, y_uni, s=22, color=cores_xy[corte], alpha=0.95, label=f"z= {corte}")

ax_30.scatter(
    ponto_eq[0], ponto_eq[1],
    s=90, marker="^", color="yellowgreen", edgecolors="black", linewidths=1.0,
    label="Última Leitura"
)

ax_30.set_title("Seção de Poincaré: plano xy.")
ax_30.set_xlabel("x")
ax_30.set_ylabel("y")
ax_30.set_xlim(-20, 20)
ax_30.set_ylim(-19, 18)
ax_30.grid(False)
ax_30.legend(loc="upper right")
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "figura_30_poincare_caos_transiente_xy.png", dpi=300)

# Exibe na tela as três figuras separadas
plt.show()
