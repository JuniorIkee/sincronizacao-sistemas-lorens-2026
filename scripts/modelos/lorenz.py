import numpy as np


def sistema_lorenz(estado, sigma, beta, rho):
    x, y, z = estado
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ])


def runge_kutta_4(funcao, estado_inicial, tempo, sigma, beta, rho):
    estados = np.zeros((len(tempo), len(estado_inicial)))
    estados[0] = estado_inicial

    for indice in range(len(tempo) - 1):
        passo = tempo[indice + 1] - tempo[indice]
        estado = estados[indice]
        k1 = funcao(estado, sigma, beta, rho)
        k2 = funcao(estado + (passo / 2) * k1, sigma, beta, rho)
        k3 = funcao(estado + (passo / 2) * k2, sigma, beta, rho)
        k4 = funcao(estado + passo * k3, sigma, beta, rho)
        estados[indice + 1] = estado + (passo / 6) * (
            k1 + 2 * k2 + 2 * k3 + k4
        )

    return estados


def calcular_secao_poincare(
    solucao, indice_variavel, valor_plano=0.0, sentido="positivo"
):
    if indice_variavel not in (0, 1, 2):
        raise ValueError("indice_variavel deve ser 0, 1 ou 2.")
    if sentido not in ("positivo", "negativo", "ambos"):
        raise ValueError("sentido deve ser 'positivo', 'negativo' ou 'ambos'.")

    pontos = []
    for indice in range(len(solucao) - 1):
        ponto_atual = solucao[indice]
        ponto_proximo = solucao[indice + 1]
        valor_atual = ponto_atual[indice_variavel] - valor_plano
        valor_proximo = ponto_proximo[indice_variavel] - valor_plano
        cruzou_positivo = valor_atual < 0 and valor_proximo >= 0
        cruzou_negativo = valor_atual > 0 and valor_proximo <= 0

        if sentido == "positivo" and not cruzou_positivo:
            continue
        if sentido == "negativo" and not cruzou_negativo:
            continue
        if sentido == "ambos" and not (cruzou_positivo or cruzou_negativo):
            continue
        if valor_proximo == valor_atual:
            continue

        alpha = -valor_atual / (valor_proximo - valor_atual)
        pontos.append(ponto_atual + alpha * (ponto_proximo - ponto_atual))

    return np.asarray(pontos)


def projetar_pontos(pontos, plano):
    indices = {"yz": (1, 2), "xz": (0, 2), "xy": (0, 1)}
    if plano not in indices:
        raise ValueError("Plano inválido. Use 'yz', 'xz' ou 'xy'.")
    if len(pontos) == 0:
        return np.array([]), np.array([])
    indice_x, indice_y = indices[plano]
    return pontos[:, indice_x], pontos[:, indice_y]


def calcular_pontos_equilibrio_lorenz(beta, rho):
    if rho <= 1:
        return []
    valor = np.sqrt(beta * (rho - 1))
    return [
        np.array([valor, valor, rho - 1]),
        np.array([-valor, -valor, rho - 1]),
    ]


def projetar_ponto_equilibrio(ponto, plano):
    x, y, z = ponto
    coordenadas = {"yz": (y, z), "xz": (x, z), "xy": (x, y)}
    if plano not in coordenadas:
        raise ValueError("Plano inválido. Use 'yz', 'xz' ou 'xy'.")
    return coordenadas[plano]
