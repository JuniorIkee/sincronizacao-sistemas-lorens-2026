import numpy as np
import matplotlib.pyplot as plt
from scripts.caminhos import preparar_pasta_imagens
from scripts.modelos.acoplamento import (
    calcular_erros_sincronizacao,
    simular_dois_lorenz,
)

PASTA_IMAGENS = preparar_pasta_imagens()

# ============================================================
# 1. PARÂMETROS DO SISTEMA DE LORENZ
# ============================================================

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0


# ============================================================
# 2. CONFIGURAÇÕES DA SIMULAÇÃO
# ============================================================

t_inicial = 0.0
t_final = 100.0
dt = 0.01

t = np.arange(t_inicial, t_final + dt, dt)


# Condições iniciais diferentes
condicao_inicial = [
    2.0, 2.0, 2.0,      # x1, y1, z1
    -1.0, -3.0, -2.0       # x2, y2, z2
]


# ============================================================
# 4. FUNÇÃO PARA EXECUTAR UMA SIMULAÇÃO
# ============================================================

def simular(K):
    tempo_simulado, estados = simular_dois_lorenz(
        t,
        condicao_inicial,
        K,
        sigma,
        rho,
        beta,
    )
    return (tempo_simulado, *estados)


# ============================================================
# 5. ESCOLHA DA FORÇA DE ACOPLAMENTO
# ============================================================

K = 5.00

tempo, x1, y1, z1, x2, y2, z2 = simular(K)


# ============================================================
# 6. ERROS DE SINCRONIZAÇÃO
# ============================================================

erro_x, erro_y, erro_z, erro_global = calcular_erros_sincronizacao(
    np.vstack((x1, y1, z1, x2, y2, z2))
)


# ============================================================
# 7. SÉRIES TEMPORAIS DE x1 E x2
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(tempo, x1, label="x1")
plt.plot(tempo, x2, label="x2", alpha=0.8)

plt.xlabel("Tempo")
plt.ylabel("x")
plt.title(f"Séries temporais - K = {K}")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PASTA_IMAGENS / f"series_temporais_K_{K}.png", dpi=300)
plt.show()


# ============================================================
# 8. ERROS EM x, y E z
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(tempo, erro_x, label="e_x = x1 - x2")
plt.plot(tempo, erro_y, label="e_y = y1 - y2")
plt.plot(tempo, erro_z, label="e_z = z1 - z2")

plt.xlabel("Tempo")
plt.ylabel("Erro")
plt.title(f"Erros de sincronização - K = {K}")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(PASTA_IMAGENS / f"erros_sincronizacao_K_{K}.png", dpi=300)
plt.tight_layout()
plt.show()


# ============================================================
# 9. ERRO GLOBAL
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(tempo, erro_global)

plt.xlabel("Tempo")
plt.ylabel("E(t)")
plt.title(f"Erro global de sincronização - K = {K}")
plt.grid(alpha=0.3)
plt.savefig(PASTA_IMAGENS / f"erros_globais_K_{K}.png", dpi=300)
plt.tight_layout()
plt.show()


# ============================================================
# 10. ERRO GLOBAL EM ESCALA LOGARÍTMICA
# ============================================================

# plt.figure(figsize=(10, 5))

# plt.semilogy(
#     tempo,
#     erro_global + 1e-16
# )

# plt.xlabel("Tempo")
# plt.ylabel("E(t)")
# plt.title(f"Erro global em escala logarítmica - K = {K}")
# plt.grid(alpha=0.3)

# plt.tight_layout()
# plt.savefig(PASTA_IMAGENS / f"erro_global_escala_logaritma_{K}.png", dpi=300)
# plt.show()


# ============================================================
# 11. ATRATORES 3D
# ============================================================

fig = plt.figure(figsize=(10, 7))

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.plot(
    x1,
    y1,
    z1,
    label="Lorenz 1",
    linewidth=0.7
)

ax.plot(
    x2,
    y2,
    z2,
    label="Lorenz 2",
    linewidth=0.7,
    alpha=0.7
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_title(
    f"Atratores de Lorenz acoplados - K = {K}"
)

ax.legend()

plt.tight_layout()
plt.savefig(PASTA_IMAGENS / f"atratores_K_{K}.png", dpi=300)
plt.show()


# ============================================================
# 12. PLANO DE SINCRONIZAÇÃO x1 x x2
# ============================================================

plt.figure(figsize=(7, 7))

plt.plot(
    x1,
    x2,
    ".",
    markersize=1
)

# reta identidade x2 = x1
limite_min = min(x1.min(), x2.min())
limite_max = max(x1.max(), x2.max())

plt.plot(
    [limite_min, limite_max],
    [limite_min, limite_max],
    "--",
    label="x2 = x1"
)

plt.xlabel("x1")
plt.ylabel("x2")

plt.title(
    f"Plano de sincronização - K = {K}"
)

plt.legend()
plt.grid(alpha=0.3)

plt.axis("equal")

plt.tight_layout()
plt.savefig(PASTA_IMAGENS / f"plano_sincronizacao_K_{K}.png", dpi=300)

plt.show()

# ============================================================
