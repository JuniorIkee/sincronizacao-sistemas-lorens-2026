import numpy as np
from scipy.integrate import solve_ivp


def lorenz_acoplado(tempo, estado, sigma, rho, beta, K):
    x1, y1, z1, x2, y2, z2 = estado

    return [
        sigma * (y1 - x1) + K * (x2 - x1),
        x1 * (rho - z1) - y1,
        x1 * y1 - beta * z1,
        sigma * (y2 - x2),
        x2 * (rho - z2) - y2,
        x2 * y2 - beta * z2,
    ]


def calcular_erros_sincronizacao(estados):
    erro_x = estados[0] - estados[3]
    erro_y = estados[1] - estados[4]
    erro_z = estados[2] - estados[5]
    erro_global = np.sqrt(erro_x**2 + erro_y**2 + erro_z**2)
    return erro_x, erro_y, erro_z, erro_global


def simular_dois_lorenz(
    tempo, estado_inicial, K, sigma, rho, beta, tempo_ativacao=None
):
    derivadas = lambda instante, estado, forca: lorenz_acoplado(
        instante, estado, sigma, rho, beta, forca
    )

    if tempo_ativacao is None:
        solucao = solve_ivp(
            derivadas,
            [tempo[0], tempo[-1]],
            estado_inicial,
            args=(K,),
            t_eval=tempo,
            method="RK45",
            rtol=1e-9,
            atol=1e-12,
        )
        if not solucao.success:
            raise RuntimeError(f"Falha na simulação de Lorenz: {solucao.message}")
        return solucao.t, solucao.y

    tempo_antes = tempo[tempo <= tempo_ativacao]
    tempo_depois = tempo[tempo >= tempo_ativacao]
    solucao_antes = solve_ivp(
        derivadas,
        [tempo[0], tempo_ativacao],
        estado_inicial,
        args=(0.0,),
        t_eval=tempo_antes,
        method="RK45",
        rtol=1e-9,
        atol=1e-12,
    )
    if not solucao_antes.success:
        raise RuntimeError(
            f"Falha na etapa sem acoplamento: {solucao_antes.message}"
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
            f"Falha na etapa com acoplamento: {solucao_depois.message}"
        )

    tempo_completo = np.concatenate((solucao_antes.t, solucao_depois.t[1:]))
    estados = np.concatenate((solucao_antes.y, solucao_depois.y[:, 1:]), axis=1)
    return tempo_completo, estados
