import numpy as np
import matplotlib.pyplot as plt
import csv
from scripts.caminhos import preparar_pasta_imagens, preparar_pasta_tabelas
from scripts.modelos.lorenz import sistema_lorenz

# ============================================================
# Configuração de saída
# ============================================================

PASTA_IMAGENS = preparar_pasta_imagens()
PASTA_TABELAS = preparar_pasta_tabelas()

# ============================================================
# Sistema de Lorenz
# ============================================================

def jacobiana_lorenz(estado, sigma, beta, rho):
    x, y, z = estado
    return np.array([
        [-sigma, sigma, 0.0],
        [rho - z, -1.0, -x],
        [y, x, -beta]
    ], dtype=float)

# ============================================================
# Método de Runge-Kutta de quarta ordem
# ============================================================

def rk4_estado(funcao, estado, h, sigma, beta, rho):
    k1 = funcao(estado, sigma, beta, rho)
    k2 = funcao(estado + (h / 2) * k1, sigma, beta, rho)
    k3 = funcao(estado + (h / 2) * k2, sigma, beta, rho)
    k4 = funcao(estado + h * k3, sigma, beta, rho)
    return estado + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def aplicar_descarte_inicial(estado_inicial, sigma, beta, rho, tempo_descarte, passo):
    """
    Integra apenas a trajetória fiducial durante o tempo de descarte.

    Esse trecho inicial não entra no cálculo dos expoentes de Lyapunov.
    A ideia é remover a influência do transiente e iniciar o cálculo já mais próximo
    do regime assintótico da trajetória.
    """
    estado = estado_inicial.astype(float).copy()
    n_passos_descarte = int(round(tempo_descarte / passo))

    for _ in range(n_passos_descarte):
        estado = rk4_estado(sistema_lorenz, estado, passo, sigma, beta, rho)

    return estado

# ============================================================
# Expoentes de Lyapunov - método do mapa tangente
# ============================================================

def derivada_estado_tangente(vetor, sigma, beta, rho):
    estado = vetor[:3]
    matriz_tangente = vetor[3:].reshape(3, 3)

    d_estado = sistema_lorenz(estado, sigma, beta, rho)
    d_matriz = jacobiana_lorenz(estado, sigma, beta, rho) @ matriz_tangente

    return np.concatenate([d_estado, d_matriz.reshape(-1)])


def rk4_estado_tangente(vetor, h, sigma, beta, rho):
    k1 = derivada_estado_tangente(vetor, sigma, beta, rho)
    k2 = derivada_estado_tangente(vetor + (h / 2) * k1, sigma, beta, rho)
    k3 = derivada_estado_tangente(vetor + (h / 2) * k2, sigma, beta, rho)
    k4 = derivada_estado_tangente(vetor + h * k3, sigma, beta, rho)
    return vetor + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def lyapunov_mapa_tangente(
    estado_inicial,
    sigma,
    beta,
    rho,
    tempo_total=100.0,
    passo=0.01,
    intervalo_qr=0.05,
    tempo_descarte=100.0,
):
    """
    Calcula a história temporal do espectro de Lyapunov pelo método do mapa tangente.

    O parâmetro tempo_descarte integra a trajetória inicial antes do cálculo, mas não
    acumula os logaritmos dos alongamentos nesse intervalo. Assim, o espectro fica
    menos contaminado pelo transiente inicial, o que é especialmente importante para
    o caso rho = 20.
    """
    estado = aplicar_descarte_inicial(
        estado_inicial, sigma, beta, rho,
        tempo_descarte=tempo_descarte,
        passo=passo,
    )

    passos_por_qr = max(1, int(round(intervalo_qr / passo)))
    n_blocos = int(round(tempo_total / (passos_por_qr * passo)))

    Q = np.eye(3)
    vetor = np.concatenate([estado, Q.reshape(-1)])
    soma_logs = np.zeros(3)

    tempos = []
    historia = []

    for bloco in range(1, n_blocos + 1):
        for _ in range(passos_por_qr):
            vetor = rk4_estado_tangente(vetor, passo, sigma, beta, rho)

        estado = vetor[:3]
        matriz_tangente = vetor[3:].reshape(3, 3)

        Q, R = np.linalg.qr(matriz_tangente)

        # Mantém a convenção de sinal para evitar saltos numéricos.
        sinais = np.sign(np.diag(R))
        sinais[sinais == 0] = 1.0
        Q = Q * sinais
        R = sinais[:, None] * R

        soma_logs += np.log(np.abs(np.diag(R)))

        # O tempo do gráfico começa em zero após o descarte inicial.
        tempo_calculo = bloco * passos_por_qr * passo
        expoentes = soma_logs / tempo_calculo

        tempos.append(tempo_calculo)
        historia.append(expoentes)

        vetor = np.concatenate([estado, Q.reshape(-1)])

    return np.array(tempos), np.array(historia)

# ============================================================
# Expoentes de Lyapunov - método da dinâmica clonada
# ============================================================

def lorenz_varios_estados(estados, sigma, beta, rho):
    """
    Versão vetorizada do sistema de Lorenz.
    Recebe uma matriz (n, 3), onde cada linha é um estado [x, y, z].
    """
    derivadas = np.empty_like(estados)
    x = estados[:, 0]
    y = estados[:, 1]
    z = estados[:, 2]

    derivadas[:, 0] = sigma * (y - x)
    derivadas[:, 1] = x * (rho - z) - y
    derivadas[:, 2] = x * y - beta * z
    return derivadas


def rk4_varios_estados(estados, h, sigma, beta, rho):
    k1 = lorenz_varios_estados(estados, sigma, beta, rho)
    k2 = lorenz_varios_estados(estados + (h / 2) * k1, sigma, beta, rho)
    k3 = lorenz_varios_estados(estados + (h / 2) * k2, sigma, beta, rho)
    k4 = lorenz_varios_estados(estados + h * k3, sigma, beta, rho)
    return estados + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def lyapunov_dinamica_clonada(
    estado_inicial,
    sigma,
    beta,
    rho,
    tempo_total=100.0,
    passo=0.01,
    intervalo_qr=0.05,
    delta=1e-8,
    tempo_descarte=100.0,
):
    """
    Calcula a história temporal do espectro de Lyapunov usando órbitas clonadas.

    Assim como no método do mapa tangente, o descarte inicial é feito antes de criar
    os clones. Desse modo, as órbitas clonadas partem de uma trajetória fiducial já
    mais próxima do regime assintótico.
    """
    estado_ref = aplicar_descarte_inicial(
        estado_inicial, sigma, beta, rho,
        tempo_descarte=tempo_descarte,
        passo=passo,
    )

    passos_por_qr = max(1, int(round(intervalo_qr / passo)))
    n_blocos = int(round(tempo_total / (passos_por_qr * passo)))

    Q = np.eye(3)

    # Linha 0: trajetória fiducial.
    # Linhas 1 a 3: clones perturbados nas direções das colunas de Q.
    estados = np.vstack([estado_ref, estado_ref + delta * Q.T])

    soma_logs = np.zeros(3)
    tempos = []
    historia = []

    for bloco in range(1, n_blocos + 1):
        for _ in range(passos_por_qr):
            estados = rk4_varios_estados(estados, passo, sigma, beta, rho)

        estado_ref = estados[0]
        clones = estados[1:]

        matriz_perturbacoes = ((clones - estado_ref).T) / delta
        Q, R = np.linalg.qr(matriz_perturbacoes)

        sinais = np.sign(np.diag(R))
        sinais[sinais == 0] = 1.0
        Q = Q * sinais
        R = sinais[:, None] * R

        soma_logs += np.log(np.abs(np.diag(R)))

        # O tempo do gráfico começa em zero após o descarte inicial.
        tempo_calculo = bloco * passos_por_qr * passo
        expoentes = soma_logs / tempo_calculo

        tempos.append(tempo_calculo)
        historia.append(expoentes)

        estados = np.vstack([estado_ref, estado_ref + delta * Q.T])

    return np.array(tempos), np.array(historia)

# ============================================================
# Utilidades para tabelas e gráficos
# ============================================================

def salvar_tabela_csv(nome_arquivo, cabecalho, linhas):
    caminho = PASTA_TABELAS / nome_arquivo
    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")
        escritor.writerow(cabecalho)
        escritor.writerows(linhas)


def salvar_tabela_markdown(nome_arquivo, cabecalho, linhas):
    caminho = PASTA_TABELAS / nome_arquivo
    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write("| " + " | ".join(cabecalho) + " |\n")
        arquivo.write("| " + " | ".join(["---"] * len(cabecalho)) + " |\n")
        for linha in linhas:
            arquivo.write("| " + " | ".join(map(str, linha)) + " |\n")


def formatar_numero(valor, casas=3):
    return f"{valor:.{casas}f}".replace(".", ",")



def criar_figura_espectro(resultados, metodo_titulo, nome_arquivo, tempo_limite=60, limites_y_por_rho=None):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    axs = axs.ravel()

    for ax, (rho, (tempos, historia)) in zip(axs, resultados.items()):
        ax.plot(tempos, historia[:, 0], color="black", label="λ1")
        ax.plot(tempos, historia[:, 1], color="blue", label="λ2")
        ax.plot(tempos, historia[:, 2], color="red", label="λ3")
        ax.axhline(0, color="black", linestyle=":", linewidth=1)
        ax.set_title(rf"$\rho = {rho}$")
        ax.set_xlabel("t[s]")
        ax.set_ylabel("λ[nats/s]")
        ax.set_xlim(0, tempo_limite)

        if limites_y_por_rho is not None and rho in limites_y_por_rho:
            ax.set_ylim(limites_y_por_rho[rho])

        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper right")

    fig.suptitle(f"Espectro de Lyapunov\n{metodo_titulo}", fontsize=14)
    plt.tight_layout()
    plt.savefig(PASTA_IMAGENS / nome_arquivo, dpi=300)
    return fig


def criar_figura_maior_expoente(resultados, metodo_titulo, nome_arquivo, tempo_limite=100):
    fig, ax = plt.subplots(figsize=(10, 6))

    for rho, (tempos, historia) in resultados.items():
        estilo = ":" if rho == 28 and "mapa" in metodo_titulo.lower() else "-"
        cor = "black" if rho == 28 and "clonada" in metodo_titulo.lower() else None
        ax.plot(tempos, historia[:, 0], linestyle=estilo, color=cor, label=rf"λ1: $\rho = {rho}$")

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"Maior expoente de Lyapunov\n{metodo_titulo}")
    ax.set_xlabel("t[s]")
    ax.set_ylabel("λ[nats/s]")
    ax.set_xlim(0, tempo_limite)
    ax.set_ylim(-1.5, 2.5)
    ax.grid(True, alpha=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PASTA_IMAGENS / nome_arquivo, dpi=300)
    return fig

# ============================================================
# Figuras 31 a 34 e Tabelas 1 a 3
# ============================================================

sigma = 10
beta = 8 / 3
estado_inicial = np.array([0.0, 1.0, 20.0])

rhos = [10, 15, 20, 28]

# Sem descarte inicial para aproximar a representação visual do artigo.
# O cálculo dos expoentes começa a partir da condição inicial definida acima.
tempo_descarte = 0.0

# Tempos de cálculo usados em cada conjunto de figuras.
# No artigo, as Figuras 31 e 33 vão até 60 s, enquanto as Figuras 32 e 34 vão até 100 s.
tempo_total_espectro = 60.0
tempo_total_maior_expoente = 100.0

# IMPORTANTE:
# Diminuir demais o intervalo_qr deixa a curva mais suave, porque a média acumulada
# é atualizada em blocos muito pequenos. Para aparecer mais variação nos primeiros
# segundos, especialmente nas Figuras 32 e 34, é melhor manter o passo pequeno para
# precisão, mas usar um intervalo_qr maior.
passo_lyap = 0.005
intervalo_qr_mt = 0.10
intervalo_qr_dc = 0.20

# Limites exatos solicitados para a Figura 31.
# A ordem no matplotlib é (mínimo, máximo), então "10 a -20" vira (-20, 10).
limites_y_figura_31 = {
    10: (-20, 10),
    15: (-25, 10),
    20: (-30, 15),
    28: (-35, 15),
}

def main():
    resultados_mt = {}
    resultados_dc = {}

    for rho in rhos:
        print(f"Calculando método do mapa tangente para rho={rho}...")
        resultados_mt[rho] = lyapunov_mapa_tangente(
            estado_inicial, sigma, beta, rho,
            tempo_total=tempo_total_maior_expoente,
            passo=passo_lyap,
            intervalo_qr=intervalo_qr_mt,
            tempo_descarte=tempo_descarte,
        )

    for rho in rhos:
        print(f"Calculando método da dinâmica clonada para rho={rho}...")
        resultados_dc[rho] = lyapunov_dinamica_clonada(
            estado_inicial, sigma, beta, rho,
            tempo_total=tempo_total_maior_expoente,
            passo=passo_lyap,
            intervalo_qr=intervalo_qr_dc,
            delta=1e-8,
            tempo_descarte=tempo_descarte,
        )

    # Figura 31
    # Sem ajuste artificial de valores iniciais: usa apenas os resultados numéricos puros.
    criar_figura_espectro(
        resultados_mt,
        "Método do mapa tangente",
        "figura_31_espectro_lyapunov_mapa_tangente.png",
        tempo_limite=tempo_total_espectro,
        limites_y_por_rho=limites_y_figura_31,
    )

    # Figura 32
    criar_figura_maior_expoente(
        resultados_mt,
        "Método do mapa tangente",
        "figura_32_maior_expoente_mapa_tangente.png",
        tempo_limite=tempo_total_maior_expoente,
    )

    # Figura 33
    criar_figura_espectro(
        resultados_dc,
        "Método da dinâmica clonada",
        "figura_33_espectro_lyapunov_dinamica_clonada.png",
        tempo_limite=tempo_total_espectro,
    )

    # Figura 34
    criar_figura_maior_expoente(
        resultados_dc,
        "Método da dinâmica clonada",
        "figura_34_maior_expoente_dinamica_clonada.png",
        tempo_limite=tempo_total_maior_expoente,
    )

    # Tabela 1: método do mapa tangente
    linhas_tabela_1 = []
    for rho in rhos:
        espectro = resultados_mt[rho][1][-1]
        linhas_tabela_1.append([
            rho,
            formatar_numero(espectro[0]),
            formatar_numero(espectro[1]),
            formatar_numero(espectro[2]),
        ])

    salvar_tabela_csv(
        "tabela_1_lyapunov_mapa_tangente.csv",
        ["rho", "lambda_1", "lambda_2", "lambda_3"],
        linhas_tabela_1,
    )
    salvar_tabela_markdown(
        "tabela_1_lyapunov_mapa_tangente.md",
        ["ρ", "λ1", "λ2", "λ3"],
        linhas_tabela_1,
    )

    # Tabela 2: método da dinâmica clonada
    linhas_tabela_2 = []
    for rho in rhos:
        espectro = resultados_dc[rho][1][-1]
        linhas_tabela_2.append([
            rho,
            formatar_numero(espectro[0]),
            formatar_numero(espectro[1]),
            formatar_numero(espectro[2]),
        ])

    salvar_tabela_csv(
        "tabela_2_lyapunov_dinamica_clonada.csv",
        ["rho", "lambda_1", "lambda_2", "lambda_3"],
        linhas_tabela_2,
    )
    salvar_tabela_markdown(
        "tabela_2_lyapunov_dinamica_clonada.md",
        ["ρ", "λ1", "λ2", "λ3"],
        linhas_tabela_2,
    )

    # Tabela 3: comparação entre os métodos
    linhas_tabela_3 = []
    for rho in rhos:
        mt = resultados_mt[rho][1][-1]
        dc = resultados_dc[rho][1][-1]
        divergencia = np.abs(mt - dc) / np.maximum(np.abs(mt), 1e-12) * 100

        linhas_tabela_3.append([
            rho,
            formatar_numero(mt[0]), formatar_numero(mt[1]), formatar_numero(mt[2]),
            formatar_numero(dc[0]), formatar_numero(dc[1]), formatar_numero(dc[2]),
            f"{divergencia[0]:.1f}%".replace(".", ","),
            f"{divergencia[1]:.1f}%".replace(".", ","),
            f"{divergencia[2]:.1f}%".replace(".", ","),
        ])

    cabecalho_tabela_3 = [
        "rho",
        "MT_lambda_1", "MT_lambda_2", "MT_lambda_3",
        "DC_lambda_1", "DC_lambda_2", "DC_lambda_3",
        "D_lambda_1", "D_lambda_2", "D_lambda_3",
    ]

    salvar_tabela_csv(
        "tabela_3_comparacao_metodos_lyapunov.csv",
        cabecalho_tabela_3,
        linhas_tabela_3,
    )
    salvar_tabela_markdown(
        "tabela_3_comparacao_metodos_lyapunov.md",
        ["ρ", "MT λ1", "MT λ2", "MT λ3", "DC λ1", "DC λ2", "DC λ3", "D λ1", "D λ2", "D λ3"],
        linhas_tabela_3,
    )

    print("\nParâmetros usados:")
    print(f"sigma = {sigma}")
    print(f"beta = {beta}")
    print(f"rhos = {rhos}")
    print(f"tempo_descarte = {tempo_descarte}")
    print("Figura 31: sem ajuste artificial de valores iniciais")
    print(f"tempo_total_espectro = {tempo_total_espectro}")
    print(f"tempo_total_maior_expoente = {tempo_total_maior_expoente}")
    print(f"passo = {passo_lyap}")
    print(f"intervalo_qr_mt = {intervalo_qr_mt}")
    print(f"intervalo_qr_dc = {intervalo_qr_dc}")
    print("\nImagens geradas em:", PASTA_IMAGENS.resolve())
    print("Tabelas geradas em:", PASTA_TABELAS.resolve())

    plt.show()


if __name__ == "__main__":
    main()
