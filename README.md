## Geradores disponíveis

Execute os comandos a partir da raiz do projeto. Cada módulo gera as figuras
e/ou tabelas descritas, sempre em `imagens/` e `tabelas/`:

```powershell
python -m scripts.gerar.unitario.secoes_poincare_lorenz_multiplos_cortes
python -m scripts.gerar.unitario.secoes_poincare_lorenz_caos_transiente
python -m scripts.gerar.unitario.expoentes_lyapunov_e_tabelas
python -m scripts.gerar.unitario.trajetorias_lorenz_sensibilidade_poincare
python -m scripts.gerar.sincronização.figuras_acoplamento_unilateral_ativacao_tardia
python -m scripts.gerar.sincronização.sincronizacao_rede_lorenz_anel
python -m scripts.gerar.sincronização.varredura_acoplamento_unilateral_ativacao_tardia
python -m scripts.gerar.sincronização.varredura_sincronizacao_rede_lorenz_anel
```
