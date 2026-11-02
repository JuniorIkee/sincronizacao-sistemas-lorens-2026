from pathlib import Path


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
PASTA_IMAGENS = RAIZ_PROJETO / "imagens"
PASTA_TABELAS = RAIZ_PROJETO / "tabelas"


def preparar_pasta_imagens():
    PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)
    return PASTA_IMAGENS


def preparar_pasta_tabelas():
    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    return PASTA_TABELAS
