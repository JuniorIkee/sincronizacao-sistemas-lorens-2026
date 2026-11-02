import numpy as np
import matplotlib.pyplot as plt
from scripts.caminhos import preparar_pasta_imagens, preparar_pasta_tabelas
from scripts.modelos.acoplamento import (
    calcular_erros_sincronizacao,
    simular_dois_lorenz,
)


PASTA_IMAGENS = preparar_pasta_imagens()
PASTA_TABELAS = preparar_pasta_tabelas()

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
TEMPO_INICIAL = 0.0
TEMPO_FINAL = 100.0
PASSO = 0.01
TEMPO_ATIVACAO = 20.0
TEMPO_INICIO_METRICAS = 40.0
CONDICAO_INICIAL = [2.0, 2.0, 2.0, -1.0, -3.0, -2.0]
VALORES_K = np.linspace(0.0, 20.0, 61)


def main():
    tempo = np.arange(TEMPO_INICIAL, TEMPO_FINAL + PASSO, PASSO)
    resultados = []

    for K in VALORES_K:
        tempo_simulado, estados = simular_dois_lorenz(
            tempo,
            CONDICAO_INICIAL,
            K,
            SIGMA,
            RHO,
            BETA,
            tempo_ativacao=TEMPO_ATIVACAO,
        )
        erro_global = calcular_erros_sincronizacao(estados)[-1]
        erro_analisado = erro_global[tempo_simulado >= TEMPO_INICIO_METRICAS]
        resultados.append((
            K,
            np.mean(erro_analisado),
            np.std(erro_analisado, ddof=1),
            np.median(erro_analisado),
            np.sqrt(np.mean(erro_analisado**2)),
            np.max(erro_analisado),
        ))

    resultados = np.asarray(resultados)
    valores_k = resultados[:, 0]
    erros_medios = resultados[:, 1]
    desvios_padrao = resultados[:, 2]

    plt.figure(figsize=(9, 5))
    plt.plot(valores_k, erros_medios, "o-", markersize=4)
    plt.xlabel("Força de acoplamento K")
    plt.ylabel("Erro médio de sincronização")
    plt.title("Erro médio em função da força de acoplamento")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PASTA_IMAGENS / "erro_medio_acoplamento.png", dpi=300)

    plt.figure(figsize=(9, 5))
    plt.plot(valores_k, erros_medios, "o-", markersize=3, label="Erro médio")
    plt.plot(
        valores_k,
        desvios_padrao,
        "s--",
        markersize=3,
        label="Desvio padrão temporal",
    )
    plt.xlabel("Força de acoplamento K")
    plt.ylabel("Erro global de sincronização")
    plt.title("Erro médio e dispersão temporal em função de K")
    plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PASTA_IMAGENS / "erro_medio_desvio_padrao_K.png", dpi=300)

    np.savetxt(
        PASTA_TABELAS / "varredura_acoplamento_unilateral_dois_lorenz.csv",
        resultados,
        delimiter=",",
        header=(
            "K,erro_medio,desvio_padrao_temporal,erro_mediano,"
            "erro_rms,erro_maximo"
        ),
        comments="",
        fmt=["%.6f", "%.12e", "%.12e", "%.12e", "%.12e", "%.12e"],
    )
    plt.show()


if __name__ == "__main__":
    main()
