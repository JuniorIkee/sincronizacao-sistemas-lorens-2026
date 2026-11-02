import numpy as np
from scipy.integrate import solve_ivp


def lorenz_anel(tempo, estado, K, quantidade=10, sigma=10.0, rho=28.0,
                 beta=8.0 / 3.0):
    estados = estado.reshape(quantidade, 3)
    x = estados[:, 0]
    y = estados[:, 1]
    z = estados[:, 2]
    acoplamento_x = (K / 2.0) * (
        (np.roll(x, 1) - x) + (np.roll(x, -1) - x)
    )
    dx = sigma * (y - x) + acoplamento_x
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.column_stack((dx, dy, dz)).ravel()


def calcular_metricas_rede(estados):
    estado_medio = np.mean(estados, axis=0)
    diferencas = estados - estado_medio[np.newaxis, :, :]
    erro_rede = np.sqrt(np.mean(np.sum(diferencas**2, axis=1), axis=0))
    desvio_x = np.std(estados[:, 0, :], axis=0, ddof=0)
    desvio_y = np.std(estados[:, 1, :], axis=0, ddof=0)
    desvio_z = np.std(estados[:, 2, :], axis=0, ddof=0)
    diferencas_vizinhos = estados - np.roll(estados, -1, axis=0)
    erro_vizinhos = np.sqrt(
        np.mean(np.sum(diferencas_vizinhos**2, axis=1), axis=0)
    )

    return {
        "estado_medio": estado_medio,
        "erro_rede": erro_rede,
        "erro_vizinhos": erro_vizinhos,
        "desvio_x": desvio_x,
        "desvio_y": desvio_y,
        "desvio_z": desvio_z,
    }


def calcular_tempo_sincronizacao(
    tempo, erro_rede, tempo_ativacao, passo, limiar=1e-3, permanencia=2.0
):
    mascara_pos_ativacao = tempo >= tempo_ativacao
    tempo_pos = tempo[mascara_pos_ativacao]
    erro_pos = erro_rede[mascara_pos_ativacao]
    quantidade_pontos = max(1, int(round(permanencia / passo)))
    abaixo_limiar = (erro_pos < limiar).astype(int)

    if abaixo_limiar.size < quantidade_pontos:
        return np.nan

    sequencias = np.convolve(
        abaixo_limiar,
        np.ones(quantidade_pontos, dtype=int),
        mode="valid",
    )
    indices = np.flatnonzero(sequencias == quantidade_pontos)
    if indices.size == 0:
        return np.nan
    return tempo_pos[indices[0]] - tempo_ativacao


def simular_anel(
    tempo,
    condicao_inicial,
    K,
    quantidade,
    sigma,
    rho,
    beta,
    tempo_ativacao=None,
):
    def derivadas(instante, estado, forca):
        return lorenz_anel(
            instante,
            estado,
            forca,
            quantidade=quantidade,
            sigma=sigma,
            rho=rho,
            beta=beta,
        )

    if tempo_ativacao is None:
        solucao = solve_ivp(
            derivadas,
            [tempo[0], tempo[-1]],
            condicao_inicial,
            args=(K,),
            t_eval=tempo,
            method="RK45",
            rtol=1e-9,
            atol=1e-12,
        )
        if not solucao.success:
            raise RuntimeError(f"Falha na simulação do anel: {solucao.message}")
        tempo_completo, estados = solucao.t, solucao.y
    else:
        tempo_antes = tempo[tempo <= tempo_ativacao]
        tempo_depois = tempo[tempo >= tempo_ativacao]
        solucao_antes = solve_ivp(
            derivadas,
            [tempo[0], tempo_ativacao],
            condicao_inicial,
            args=(0.0,),
            t_eval=tempo_antes,
            method="RK45",
            rtol=1e-9,
            atol=1e-12,
        )
        if not solucao_antes.success:
            raise RuntimeError(
                f"Falha na etapa sem acoplamento do anel: {solucao_antes.message}"
            )
        solucao_depois = solve_ivp(
            derivadas,
            [tempo_ativacao, tempo[-1]],
            solucao_antes.y[:, -1],
            args=(K,),
            t_eval=tempo_depois,
            method="RK45",
            rtol=1e-9,
            atol=1e-12,
        )
        if not solucao_depois.success:
            raise RuntimeError(
                f"Falha na etapa com acoplamento do anel: {solucao_depois.message}"
            )
        tempo_completo = np.concatenate(
            (solucao_antes.t, solucao_depois.t[1:])
        )
        estados = np.concatenate(
            (solucao_antes.y, solucao_depois.y[:, 1:]), axis=1
        )

    return tempo_completo, estados.reshape(quantidade, 3, -1)
