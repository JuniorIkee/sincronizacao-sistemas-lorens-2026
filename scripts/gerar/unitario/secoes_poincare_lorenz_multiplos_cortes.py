import numpy as np
import matplotlib.pyplot as plt
from scripts.caminhos import preparar_pasta_imagens
from scripts.modelos.lorenz import (
    calcular_secao_poincare,
    projetar_pontos,
    runge_kutta_4,
    sistema_lorenz,
)

# Pasta onde as imagens serão salvas.
PASTA_SAIDA = preparar_pasta_imagens()


# ============================================================
# Parâmetros usados nas Figuras 25, 26 e 27
# ============================================================

sigma = 10
beta = 8 / 3
rho = 28

passo = 0.01
tempo_poincare = np.arange(0, 200, passo)

# Mantém t = 0, como nas figuras de validação.
tempo_transiente = 0
indice_transiente = int(tempo_transiente / passo)

estado_inicial = np.array([0.0, 1.0, 20.0])

solucao = runge_kutta_4(
    sistema_lorenz,
    estado_inicial,
    tempo_poincare,
    sigma,
    beta,
    rho
)

solucao = solucao[indice_transiente:]

# Figuras 25 e 26
cortes_unilateral = [-10, -5, 0, 5, 10]
cortes_bilateral = [0, 5, -5, 10, -10, 15, -15]

cores_unilateral = {
    -10: "blue",
    -5: "lightgreen",
    0: "black",
    5: "green",
    10: "navy",
}

cores_bilateral = {
    0: "black",
    5: "blue",
    -5: "blue",
    10: "green",
    -10: "green",
    15: "y",
    -15: "y",
}

# Figura 27
cortes_xy = [10, 20, 25, 30]
cores_xy = {
    10: "c",
    20: "red",
    25: "m",
    30: "black",
}

cores_cinza = {corte: "gray" for corte in cortes_bilateral}
cores_cinza_xy = {corte: "gray" for corte in cortes_xy}


# ============================================================
# Funções auxiliares de plotagem
# ============================================================

def plotar_cortes_yz(eixo, cortes, sentido, cores, alpha=1.0, tamanho=10, rotulo=True):
    for corte in cortes:
        pontos = calcular_secao_poincare(solucao, 0, corte, sentido)
        y, z = projetar_pontos(pontos, "yz")
        eixo.scatter(y, z, s=tamanho, color=cores[corte], alpha=alpha,
                     label=f"x= {corte}" if rotulo else None)


def plotar_cortes_xz(eixo, cortes, sentido, cores, alpha=1.0, tamanho=10, rotulo=True):
    for corte in cortes:
        pontos = calcular_secao_poincare(solucao, 1, corte, sentido)
        x, z = projetar_pontos(pontos, "xz")
        eixo.scatter(x, z, s=tamanho, color=cores[corte], alpha=alpha,
                     label=f"y= {corte}" if rotulo else None)


def plotar_cortes_xy(eixo, cortes, sentido, cores, alpha=1.0, tamanho=10, rotulo=True):
    for corte in cortes:
        pontos = calcular_secao_poincare(solucao, 2, corte, sentido)
        x, y = projetar_pontos(pontos, "xy")
        eixo.scatter(x, y, s=tamanho, color=cores[corte], alpha=alpha,
                     label=f"z= {corte}" if rotulo else None)


# ============================================================
# Figura 25 - plano yz
# ============================================================

fig_25a, ax_25a = plt.subplots(figsize=(12, 5))
plotar_cortes_yz(ax_25a, cortes_unilateral, "positivo", cores_unilateral, alpha=0.9, tamanho=9)
ax_25a.set_title("Seção de Poincaré: plano yz.")
ax_25a.set_xlabel("y")
ax_25a.set_ylabel("z")
ax_25a.set_xlim(-20, 20)
ax_25a.set_ylim(2, 45)
ax_25a.grid(False)
ax_25a.legend(loc="upper right")
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "figura_25a_poincare_yz_unilateral.png", dpi=300)
plt.close(fig_25a)
plt.close(fig_25a)

fig_25b, ax_25b = plt.subplots(figsize=(12, 5))
plotar_cortes_yz(ax_25b, cortes_bilateral, "ambos", cores_bilateral, alpha=0.8, tamanho=9)
ax_25b.set_title("Pontos no plano yz")
ax_25b.set_xlabel("y")
ax_25b.set_ylabel("z")
ax_25b.set_xlim(-25, 30)
ax_25b.set_ylim(2, 47)
ax_25b.grid(False)
ax_25b.legend(loc="upper right")
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "figura_25b_poincare_yz_bilateral.png", dpi=300)
plt.close(fig_25b)
plt.close(fig_25b)

fig_25c, ax_25c = plt.subplots(figsize=(12, 5))
plotar_cortes_yz(ax_25c, cortes_bilateral, "ambos", cores_cinza, alpha=0.45, tamanho=8, rotulo=False)
plotar_cortes_yz(ax_25c, cortes_unilateral, "positivo", cores_unilateral, alpha=0.9, tamanho=9)
ax_25c.set_title("Seção de Poincaré: plano yz - composição.")
ax_25c.set_xlabel("y")
ax_25c.set_ylabel("z")
ax_25c.set_xlim(-20, 20)
ax_25c.set_ylim(2, 47)
ax_25c.grid(False)
ax_25c.legend(loc="upper right")
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "figura_25c_poincare_yz_composicao.png", dpi=300)
plt.close(fig_25c)
plt.close(fig_25c)

fig_total_25, eixos_25 = plt.subplots(nrows=3, ncols=1, figsize=(12, 15))
plotar_cortes_yz(eixos_25[0], cortes_unilateral, "positivo", cores_unilateral, alpha=0.9, tamanho=9)
eixos_25[0].set_title("Seção de Poincaré: plano yz.")
eixos_25[0].set_xlabel("y")
eixos_25[0].set_ylabel("z")
eixos_25[0].set_xlim(-20, 20)
eixos_25[0].set_ylim(2, 45)
eixos_25[0].legend(loc="upper right")

plotar_cortes_yz(eixos_25[1], cortes_bilateral, "ambos", cores_bilateral, alpha=0.8, tamanho=9)
eixos_25[1].set_title("Pontos no plano yz")
eixos_25[1].set_xlabel("y")
eixos_25[1].set_ylabel("z")
eixos_25[1].set_xlim(-25, 30)
eixos_25[1].set_ylim(2, 47)
eixos_25[1].legend(loc="upper right")

plotar_cortes_yz(eixos_25[2], cortes_bilateral, "ambos", cores_cinza, alpha=0.45, tamanho=8, rotulo=False)
plotar_cortes_yz(eixos_25[2], cortes_unilateral, "positivo", cores_unilateral, alpha=0.9, tamanho=9)
eixos_25[2].set_title("Seção de Poincaré: plano yz - composição.")
eixos_25[2].set_xlabel("y")
eixos_25[2].set_ylabel("z")
eixos_25[2].set_xlim(-20, 20)
eixos_25[2].set_ylim(2, 47)
eixos_25[2].legend(loc="upper right")

for eixo in eixos_25:
    eixo.grid(False)

plt.tight_layout()
plt.savefig(PASTA_SAIDA / "figura_25_poincare_yz_completa.png", dpi=300)


# ============================================================
# Figura 26 - plano xz
# ============================================================

fig_26a, ax_26a = plt.subplots(figsize=(12, 5))
plotar_cortes_xz(ax_26a, cortes_unilateral, "positivo", cores_unilateral, alpha=0.9, tamanho=9)
ax_26a.set_title("Seção de Poincaré: plano xz.")
ax_26a.set_xlabel("x")
ax_26a.set_ylabel("z")
ax_26a.set_xlim(-20, 20)
ax_26a.set_ylim(2, 47)
ax_26a.grid(False)
ax_26a.legend(loc="upper right")
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "figura_26a_poincare_xz_unilateral.png", dpi=300)
plt.close(fig_26a)
plt.close(fig_26a)

fig_26b, ax_26b = plt.subplots(figsize=(12, 5))
plotar_cortes_xz(ax_26b, cortes_bilateral, "ambos", cores_bilateral, alpha=0.8, tamanho=9)
ax_26b.set_title("Pontos no plano xz.")
ax_26b.set_xlabel("x")
ax_26b.set_ylabel("z")
ax_26b.set_xlim(-18, 18)
ax_26b.set_ylim(2, 47)
ax_26b.grid(False)
ax_26b.legend(loc="upper right")
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "figura_26b_poincare_xz_bilateral.png", dpi=300)
plt.close(fig_26b)
plt.close(fig_26b)

fig_26c, ax_26c = plt.subplots(figsize=(12, 5))
plotar_cortes_xz(ax_26c, cortes_bilateral, "ambos", cores_cinza, alpha=0.45, tamanho=8, rotulo=False)
plotar_cortes_xz(ax_26c, cortes_unilateral, "positivo", cores_unilateral, alpha=0.9, tamanho=9)
ax_26c.set_title("Seção de Poincaré: plano xz - composição.")
ax_26c.set_xlabel("x")
ax_26c.set_ylabel("z")
ax_26c.set_xlim(-20, 20)
ax_26c.set_ylim(2, 47)
ax_26c.grid(False)
ax_26c.legend(loc="upper right")
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "figura_26c_poincare_xz_composicao.png", dpi=300)
plt.close(fig_26c)
plt.close(fig_26c)

fig_total_26, eixos_26 = plt.subplots(nrows=3, ncols=1, figsize=(12, 15))
plotar_cortes_xz(eixos_26[0], cortes_unilateral, "positivo", cores_unilateral, alpha=0.9, tamanho=9)
eixos_26[0].set_title("Seção de Poincaré: plano xz.")
eixos_26[0].set_xlabel("x")
eixos_26[0].set_ylabel("z")
eixos_26[0].set_xlim(-20, 20)
eixos_26[0].set_ylim(2, 47)
eixos_26[0].legend(loc="upper right")

plotar_cortes_xz(eixos_26[1], cortes_bilateral, "ambos", cores_bilateral, alpha=0.8, tamanho=9)
eixos_26[1].set_title("Pontos no plano xz.")
eixos_26[1].set_xlabel("x")
eixos_26[1].set_ylabel("z")
eixos_26[1].set_xlim(-18, 18)
eixos_26[1].set_ylim(2, 47)
eixos_26[1].legend(loc="upper right")

plotar_cortes_xz(eixos_26[2], cortes_bilateral, "ambos", cores_cinza, alpha=0.45, tamanho=8, rotulo=False)
plotar_cortes_xz(eixos_26[2], cortes_unilateral, "positivo", cores_unilateral, alpha=0.9, tamanho=9)
eixos_26[2].set_title("Seção de Poincaré: plano xz - composição.")
eixos_26[2].set_xlabel("x")
eixos_26[2].set_ylabel("z")
eixos_26[2].set_xlim(-20, 20)
eixos_26[2].set_ylim(2, 47)
eixos_26[2].legend(loc="upper right")

for eixo in eixos_26:
    eixo.grid(False)

plt.tight_layout()
plt.savefig(PASTA_SAIDA / "figura_26_poincare_xz_completa.png", dpi=300)


# ============================================================
# Figura 27 - plano xy
# ============================================================

fig_27a, ax_27a = plt.subplots(figsize=(10, 5))
plotar_cortes_xy(ax_27a, cortes_xy, "positivo", cores_xy, alpha=0.9, tamanho=9)
ax_27a.set_title("Seção de Poincaré: Plano xy.")
ax_27a.set_xlabel("x")
ax_27a.set_ylabel("y")
ax_27a.set_xlim(-20, 20)
ax_27a.set_ylim(-28, 28)
ax_27a.grid(False)
ax_27a.legend(loc="upper right")
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "figura_27a_poincare_xy_unilateral.png", dpi=300)
plt.close(fig_27a)
plt.close(fig_27a)

fig_27b, ax_27b = plt.subplots(figsize=(10, 5))
plotar_cortes_xy(ax_27b, cortes_xy, "ambos", cores_xy, alpha=0.8, tamanho=9)
ax_27b.set_title("Pontos no plano xy.")
ax_27b.set_xlabel("x")
ax_27b.set_ylabel("y")
ax_27b.set_xlim(-18, 18)
ax_27b.set_ylim(-28, 28)
ax_27b.grid(False)
ax_27b.legend(loc="upper right")
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "figura_27b_poincare_xy_bilateral.png", dpi=300)
plt.close(fig_27b)
plt.close(fig_27b)

fig_27c, ax_27c = plt.subplots(figsize=(10, 5))
plotar_cortes_xy(ax_27c, cortes_xy, "ambos", cores_cinza_xy, alpha=0.45, tamanho=8, rotulo=False)
plotar_cortes_xy(ax_27c, cortes_xy, "positivo", cores_xy, alpha=0.9, tamanho=9)
ax_27c.set_title("Seção de Poincaré: plano xy - composição.")
ax_27c.set_xlabel("x")
ax_27c.set_ylabel("y")
ax_27c.set_xlim(-20, 20)
ax_27c.set_ylim(-28, 28)
ax_27c.grid(False)
ax_27c.legend(loc="upper right")
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "figura_27c_poincare_xy_composicao.png", dpi=300)
plt.close(fig_27c)
plt.close(fig_27c)

fig_total_27, eixos_27 = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
plotar_cortes_xy(eixos_27[0], cortes_xy, "positivo", cores_xy, alpha=0.9, tamanho=9)
eixos_27[0].set_title("Seção de Poincaré: Plano xy.")
eixos_27[0].set_xlabel("x")
eixos_27[0].set_ylabel("y")
eixos_27[0].set_xlim(-20, 20)
eixos_27[0].set_ylim(-28, 28)
eixos_27[0].legend(loc="upper right")

plotar_cortes_xy(eixos_27[1], cortes_xy, "ambos", cores_xy, alpha=0.8, tamanho=9)
eixos_27[1].set_title("Pontos no plano xy.")
eixos_27[1].set_xlabel("x")
eixos_27[1].set_ylabel("y")
eixos_27[1].set_xlim(-18, 18)
eixos_27[1].set_ylim(-28, 28)
eixos_27[1].legend(loc="upper right")

plotar_cortes_xy(eixos_27[2], cortes_xy, "ambos", cores_cinza_xy, alpha=0.45, tamanho=8, rotulo=False)
plotar_cortes_xy(eixos_27[2], cortes_xy, "positivo", cores_xy, alpha=0.9, tamanho=9)
eixos_27[2].set_title("Seção de Poincaré: plano xy - composição.")
eixos_27[2].set_xlabel("x")
eixos_27[2].set_ylabel("y")
eixos_27[2].set_xlim(-20, 20)
eixos_27[2].set_ylim(-28, 28)
eixos_27[2].legend(loc="upper right")

for eixo in eixos_27:
    eixo.grid(False)

plt.tight_layout()
plt.savefig(PASTA_SAIDA / "figura_27_poincare_xy_completa.png", dpi=300)

# Exibe somente as figuras agrupadas na tela.
plt.show()
