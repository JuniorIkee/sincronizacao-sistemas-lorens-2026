import numpy as np
import matplotlib.pyplot as plt
from scripts.caminhos import preparar_pasta_imagens, preparar_pasta_tabelas
from scripts.modelos.anel import (
    calcular_metricas_rede,
    calcular_tempo_sincronizacao,
    simular_anel,
)


PASTA_IMAGENS = preparar_pasta_imagens()
PASTA_TABELAS = preparar_pasta_tabelas()

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
QUANTIDADE_OSCILADORES = 10
TEMPO_INICIAL = 0.0
TEMPO_FINAL = 100.0
PASSO = 0.01
TEMPO_ATIVACAO = 20.0
TEMPO_INICIO_METRICAS = 40.0
VALORES_K = np.linspace(0.0, 160.0, 82)


def main():
    tempo = np.arange(TEMPO_INICIAL, TEMPO_FINAL + PASSO, PASSO)
    gerador = np.random.default_rng(seed=42)
    condicoes_iniciais = np.column_stack((
        gerador.uniform(-20.0, 20.0, QUANTIDADE_OSCILADORES),
        gerador.uniform(-30.0, 30.0, QUANTIDADE_OSCILADORES),
        gerador.uniform(0.0, 50.0, QUANTIDADE_OSCILADORES),
    ))
    condicao_inicial = condicoes_iniciais.ravel()
    resultados = []

    for K in VALORES_K:
        tempo_simulado, estados = simular_anel(
            tempo,
            condicao_inicial,
            K,
            QUANTIDADE_OSCILADORES,
            SIGMA,
            RHO,
            BETA,
            tempo_ativacao=TEMPO_ATIVACAO,
        )
        metricas = calcular_metricas_rede(estados)
        erro_rede = metricas["erro_rede"]
        erro_analisado = erro_rede[tempo_simulado >= TEMPO_INICIO_METRICAS]
        resultados.append((
            K,
            np.mean(erro_analisado),
            np.std(erro_analisado, ddof=1),
            np.median(erro_analisado),
            np.sqrt(np.mean(erro_analisado**2)),
            np.max(erro_analisado),
            calcular_tempo_sincronizacao(
                tempo_simulado, erro_rede, TEMPO_ATIVACAO, PASSO
            ),
        ))

    resultados = np.asarray(resultados)
    valores_k = resultados[:, 0]
    erros_medios = resultados[:, 1]
    desvios_temporais = resultados[:, 2]
    tempos_sincronizacao = resultados[:, 6]

    plt.figure(figsize=(9, 5))
    plt.semilogy(valores_k, erros_medios + 1e-16, "o-", markersize=4,
                 label="Erro médio")
    plt.semilogy(valores_k, desvios_temporais + 1e-16, "s--", markersize=4,
                 label="Desvio padrão temporal")
    plt.xlabel("Força de acoplamento K")
    plt.ylabel("Erro global da rede")
    plt.title(f"Sincronização do anel com {QUANTIDADE_OSCILADORES} osciladores")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PASTA_IMAGENS / "varredura_erro_desvio_padrao_anel.png", dpi=300)

    mascara_sincronizados = np.isfinite(tempos_sincronizacao)
    plt.figure(figsize=(9, 5))
    plt.plot(
        valores_k[mascara_sincronizados],
        tempos_sincronizacao[mascara_sincronizados],
        "o-",
        markersize=4,
    )
    plt.xlabel("Força de acoplamento K")
    plt.ylabel("Tempo para sincronizar após ligar o anel")
    plt.title("Tempo de sincronização, limiar de erro = 10⁻³")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PASTA_IMAGENS / "tempo_sincronizacao_por_K.png", dpi=300)

    np.savetxt(
        PASTA_TABELAS / "varredura_sincronizacao_rede_lorenz_anel.csv",
        resultados,
        delimiter=",",
        header=(
            "K,erro_medio,desvio_padrao_temporal,erro_mediano,erro_rms,"
            "erro_maximo,tempo_sincronizacao_apos_ativacao"
        ),
        comments="",
        fmt=["%.6f", "%.12e", "%.12e", "%.12e", "%.12e", "%.12e", "%.6f"],
    )
    plt.show()


if __name__ == "__main__":
    main()
