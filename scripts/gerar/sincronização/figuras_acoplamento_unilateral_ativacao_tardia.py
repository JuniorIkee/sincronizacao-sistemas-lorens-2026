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

# Os osciladores evoluem sem acoplamento até este instante.
# A partir daqui, a força K escolhida é ativada.
tempo_ativacao_acoplamento = 20.0
LIMIAR_SINCRONIZACAO = 1e-3
PERMANENCIA_SINCRONIZACAO = 2.0

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
        tempo_ativacao=tempo_ativacao_acoplamento,
    )
    return (tempo_simulado, *estados)


def marcar_ativacao_acoplamento():
    """Destaca, nos gráficos temporais, quando o acoplamento é ligado."""
    plt.axvspan(
        t_inicial,
        tempo_ativacao_acoplamento,
        color="gray",
        alpha=0.12,
        label="Sem acoplamento"
    )
    plt.axvline(
        tempo_ativacao_acoplamento,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Acoplamento ligado (t = {tempo_ativacao_acoplamento:g} s)"
    )


def encontrar_tempo_sincronizacao(tempo, erro):
    """Retorna o primeiro instante em que o erro permanece abaixo do limiar."""
    mascara_pos_ativacao = tempo >= tempo_ativacao_acoplamento
    tempo_pos_ativacao = tempo[mascara_pos_ativacao]
    erro_pos_ativacao = erro[mascara_pos_ativacao]
    pontos_permanencia = max(1, int(round(PERMANENCIA_SINCRONIZACAO / dt)))
    abaixo_limiar = erro_pos_ativacao < LIMIAR_SINCRONIZACAO

    if abaixo_limiar.size < pontos_permanencia:
        return None

    permanencia = np.convolve(
        abaixo_limiar.astype(int),
        np.ones(pontos_permanencia, dtype=int),
        mode="valid",
    )
    indices = np.flatnonzero(permanencia == pontos_permanencia)
    if indices.size == 0:
        return None
    return tempo_pos_ativacao[indices[0]]


def marcar_sincronizacao(tempo_sincronizacao):
    """Marca o início da sincronização operacional nos gráficos temporais."""
    if tempo_sincronizacao is None:
        return

    plt.axvspan(
        tempo_sincronizacao,
        t_final,
        color="seagreen",
        alpha=0.08,
        label=f"Sincronizado (E < {LIMIAR_SINCRONIZACAO:g})",
    )
    plt.axvline(
        tempo_sincronizacao,
        color="seagreen",
        linestyle=":",
        linewidth=1.6,
        label=f"Início da sincronização (t = {tempo_sincronizacao:.2f})",
    )


# ============================================================
# 5. ESCOLHA DA FORÇA DE ACOPLAMENTO
# ============================================================

# K = 10 mostra a convergência após a ativação do acoplamento.
K = 10.0

tempo, x1, y1, z1, x2, y2, z2 = simular(K)


# ============================================================
# 6. ERROS DE SINCRONIZAÇÃO
# ============================================================

erro_x, erro_y, erro_z, erro_global = calcular_erros_sincronizacao(
    np.vstack((x1, y1, z1, x2, y2, z2))
)
tempo_sincronizacao = encontrar_tempo_sincronizacao(tempo, erro_global)
if tempo_sincronizacao is None:
    print("O critério de sincronização não foi atingido nesta simulação.")
else:
    print(
        f"Sincronização indicada em t = {tempo_sincronizacao:.2f}: "
        f"E(t) < {LIMIAR_SINCRONIZACAO:g} por "
        f"{PERMANENCIA_SINCRONIZACAO:g} unidades de tempo."
    )


# ============================================================
# 7. SÉRIES TEMPORAIS DE x1 E x2
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(tempo, x1, label="x1")
plt.plot(tempo, x2, label="x2", alpha=0.8)
marcar_ativacao_acoplamento()
marcar_sincronizacao(tempo_sincronizacao)

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
marcar_ativacao_acoplamento()
marcar_sincronizacao(tempo_sincronizacao)

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

plt.semilogy(
    tempo,
    np.maximum(erro_global, np.finfo(float).tiny),
    label="Erro global E(t)",
)
plt.axhline(
    LIMIAR_SINCRONIZACAO,
    color="firebrick",
    linestyle="--",
    linewidth=1.2,
    label=f"Limiar de sincronização ({LIMIAR_SINCRONIZACAO:g})",
)
marcar_ativacao_acoplamento()
marcar_sincronizacao(tempo_sincronizacao)

plt.xlabel("Tempo")
plt.ylabel("Erro global E(t) (escala logarítmica)")
plt.title(f"Erro global de sincronização após ativação tardia - K = {K}")
plt.legend()
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

mascara_antes = tempo <= tempo_ativacao_acoplamento
mascara_depois = tempo >= tempo_ativacao_acoplamento

ax.plot(x1[mascara_antes], y1[mascara_antes], z1[mascara_antes],
        label="Lorenz 1 sem acoplamento", linewidth=0.7, alpha=0.45)
ax.plot(x2[mascara_antes], y2[mascara_antes], z2[mascara_antes],
        label="Lorenz 2 sem acoplamento", linewidth=0.7, alpha=0.45)
ax.plot(x1[mascara_depois], y1[mascara_depois], z1[mascara_depois],
        label="Lorenz 1 após ativação", linewidth=0.8)
ax.plot(x2[mascara_depois], y2[mascara_depois], z2[mascara_depois],
        label="Lorenz 2 após ativação", linewidth=0.8, alpha=0.8)

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

plt.plot(x1[mascara_antes], x2[mascara_antes], ".", markersize=1,
         alpha=0.35, label="Sem acoplamento")
plt.plot(x1[mascara_depois], x2[mascara_depois], ".", markersize=1,
         alpha=0.65, label="Após ativação")

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
