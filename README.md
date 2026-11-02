# Sincronização de sistemas de Lorenz

Os modelos e funções de cálculo ficam em `scripts/modelos/`. Os geradores ficam
separados em `scripts/gerar/unitario/` e `scripts/gerar/sincronização/`. Os
resultados são gravados em `imagens/` e `tabelas/`, independentemente do
diretório de onde o comando é executado.

## Instalação

```powershell
python -m pip install -r requirements.txt
```

## Geradores disponíveis

Execute os comandos a partir da raiz do projeto. Cada módulo gera as figuras
e/ou tabelas descritas, sempre em `imagens/` e `tabelas/`:

```powershell
python -m scripts.gerar.unitario.secoes_poincare_lorenz_multiplos_cortes
python -m scripts.gerar.unitario.secoes_poincare_lorenz_caos_transiente
python -m scripts.gerar.unitario.expoentes_lyapunov_e_tabelas
python -m scripts.gerar.unitario.trajetorias_lorenz_sensibilidade_poincare
python -m scripts.gerar.sincronização.figuras_acoplamento_unilateral_ativacao_tardia
python -m scripts.gerar.sincronização.figuras_acoplamento_unilateral_sem_ativacao
python -m scripts.gerar.sincronização.sincronizacao_rede_lorenz_anel
python -m scripts.gerar.sincronização.varredura_acoplamento_unilateral_ativacao_tardia
python -m scripts.gerar.sincronização.varredura_acoplamento_unilateral_sem_ativacao
python -m scripts.gerar.sincronização.varredura_sincronizacao_rede_lorenz_anel
```

- `secoes_poincare_lorenz_multiplos_cortes`: seções nos planos yz, xz e xy,
  com múltiplos cortes.
- `secoes_poincare_lorenz_caos_transiente`: seções de Poincaré do regime de
  caos transiente.
- `expoentes_lyapunov_e_tabelas`: espectros e maiores expoentes de Lyapunov,
  além das tabelas de comparação entre métodos.
- `trajetorias_lorenz_sensibilidade_poincare`: comparação de trajetórias para
  diferentes parâmetros/condições iniciais e figura de cortes de Poincaré.
- `figuras_acoplamento_unilateral_ativacao_tardia`: figuras de uma simulação
  dos dois sistemas antes e depois de ativar o acoplamento. Os gráficos
  temporais indicam quando o erro global permanece abaixo de `1e-3` por
  `2` unidades de tempo; no exemplo com `K = 10`, isso ocorre em `t = 40.55`.
- `figuras_acoplamento_unilateral_sem_ativacao`: figuras de uma simulação com
  o acoplamento ligado desde o início.
- `sincronizacao_rede_lorenz_anel`: figuras de uma simulação da rede de Lorenz
  em anel.
- `varredura_acoplamento_unilateral_ativacao_tardia`: varredura de K para os
  dois sistemas com ativação tardia do acoplamento.
- `varredura_acoplamento_unilateral_sem_ativacao`: varredura de K com o
  acoplamento ligado desde o início.
- `varredura_sincronizacao_rede_lorenz_anel`: varredura de K da rede de Lorenz
  em anel.

Os arquivos de `documentos/` são ignorados pelo Git. Resultados gerados podem
ser recriados executando os programas correspondentes.
