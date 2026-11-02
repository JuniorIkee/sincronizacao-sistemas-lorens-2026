import numpy as np
import matplotlib.pyplot as plt
from scripts.caminhos import preparar_pasta_imagens
from scripts.modelos.anel import (
    calcular_metricas_rede,
    lorenz_anel as calcular_derivadas_anel,
    simular_anel,
)


# ============================================================
# 1. CONFIGURAÇÃO GERAL
# ============================================================

PASTA_IMAGENS = preparar_pasta_imagens()

# Parâmetros clássicos do sistema de Lorenz
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Quantidade de osciladores no anel
N = 10

# Configuração temporal
t_inicial = 0.0
t_final = 100.0
dt = 0.01
t = np.arange(t_inicial, t_final + dt, dt)

# Os sistemas evoluem separadamente até 20 s.
tempo_ativacao_acoplamento = 20.0

# Força usada nos gráficos individuais.
# O anel costuma exigir K maior do que o sistema com apenas dois Lorenz.
K_exemplo = 60.0

# O resultado é reproduzível sempre que o programa é executado.
gerador = np.random.default_rng(seed=42)

# Uma condição inicial diferente para cada Lorenz.
condicoes_iniciais = np.column_stack((
    gerador.uniform(-20.0, 20.0, N),   # valores de x
    gerador.uniform(-30.0, 30.0, N),   # valores de y
    gerador.uniform(0.0, 50.0, N)      # valores de z
))

# O integrador recebe um vetor com 3N posições.
condicao_inicial = condicoes_iniciais.ravel()

# ============================================================
# 2. REDE DE LORENZ EM ANEL BIDIRECIONAL
# ============================================================

def lorenz_anel(tempo, estado, K):
    return calcular_derivadas_anel(
        tempo, estado, K, quantidade=N, sigma=sigma, rho=rho, beta=beta
    )


# ============================================================
# 3. SIMULAÇÃO EM DUAS ETAPAS
# ============================================================

def simular(K):
    return simular_anel(
        t,
        condicao_inicial,
        K,
        N,
        sigma,
        rho,
        beta,
        tempo_ativacao=tempo_ativacao_acoplamento,
    )


# ============================================================
# 4. MÉTRICAS DA REDE
# ============================================================

def marcar_ativacao():
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
        label=f"Anel ligado (t = {tempo_ativacao_acoplamento:g} s)"
    )


# ============================================================
# 5. SIMULAÇÃO PRINCIPAL
# ============================================================

tempo, estados = simular(K_exemplo)
metricas = calcular_metricas_rede(estados)

x = estados[:, 0, :]
y = estados[:, 1, :]
z = estados[:, 2, :]


# ============================================================
# 6. SÉRIES TEMPORAIS DOS DEZ OSCILADORES
# ============================================================

plt.figure(figsize=(12, 6))

for i in range(N):
    plt.plot(tempo, x[i], linewidth=0.8, alpha=0.75,
             label=f"x{i + 1}")

plt.plot(
    tempo,
    metricas["estado_medio"][0],
    color="black",
    linewidth=1.4,
    label="Média da rede"
)

marcar_ativacao()
plt.xlabel("Tempo")
plt.ylabel("x")
plt.title(f"Séries temporais dos {N} Lorenz em anel, K = {K_exemplo:g}")
plt.grid(alpha=0.3)
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.savefig(PASTA_IMAGENS / "series_temporais_anel.png", dpi=300)
plt.show()


# ============================================================
# 7. ERROS GLOBAIS DA REDE E ENTRE VIZINHOS
# ============================================================

plt.figure(figsize=(10, 5))
plt.semilogy(tempo, metricas["erro_rede"] + 1e-16,
             label="Erro em relação à média")
plt.semilogy(tempo, metricas["erro_vizinhos"] + 1e-16,
             label="Erro entre vizinhos", alpha=0.8)
marcar_ativacao()
plt.xlabel("Tempo")
plt.ylabel("Erro de sincronização")
plt.title(f"Sincronização global do anel, K = {K_exemplo:g}")
plt.grid(alpha=0.3, which="both")
plt.legend()
plt.tight_layout()
plt.savefig(PASTA_IMAGENS / "erro_global_anel.png", dpi=300)
plt.show()


# ============================================================
# 8. DESVIO PADRÃO ENTRE OS OSCILADORES
# ============================================================

plt.figure(figsize=(10, 5))
plt.semilogy(tempo, metricas["desvio_x"] + 1e-16, label="Desvio padrão de x")
plt.semilogy(tempo, metricas["desvio_y"] + 1e-16, label="Desvio padrão de y")
plt.semilogy(tempo, metricas["desvio_z"] + 1e-16, label="Desvio padrão de z")
marcar_ativacao()
plt.xlabel("Tempo")
plt.ylabel("Desvio padrão entre osciladores")
plt.title(f"Dispersão dos estados da rede, K = {K_exemplo:g}")
plt.grid(alpha=0.3, which="both")
plt.legend()
plt.tight_layout()
plt.savefig(PASTA_IMAGENS / "desvios_padrao_anel.png", dpi=300)
plt.show()


# ============================================================
# 9. MAPA DE CALOR DA VARIÁVEL x
# ============================================================

plt.figure(figsize=(12, 5))
imagem = plt.imshow(
    x,
    aspect="auto",
    origin="lower",
    extent=[tempo[0], tempo[-1], 0.5, N + 0.5],
    cmap="coolwarm"
)
plt.axvline(
    tempo_ativacao_acoplamento,
    color="black",
    linestyle="--",
    linewidth=1.2
)
plt.colorbar(imagem, label="Valor de x")
plt.xlabel("Tempo")
plt.ylabel("Oscilador")
plt.yticks(np.arange(1, N + 1))
plt.title(f"Mapa de calor da sincronização do anel, K = {K_exemplo:g}")
plt.tight_layout()
plt.savefig(PASTA_IMAGENS / "mapa_calor_x_anel.png", dpi=300)
plt.show()


# ============================================================
