@"
# Agente DQN para CartPole-v1

## Objetivo

Desenvolver um agente baseado em Deep Q-Network (DQN) capaz de jogar autonomamente o ambiente CartPole-v1, utilizando Python, TensorFlow e Gymnasium.

## Ambiente

O CartPole-v1 possui:

- Estado com 4 variáveis;
- 2 ações possíveis;
- Recompensa de 1 ponto por passo;
- Limite máximo usual de 500 pontos por episódio.

## Arquivos principais

- `teste_ambiente.py`: verifica se o Gymnasium está funcionando.
- `treino_dqn.py`: cria e treina o agente DQN.
- `testar_agente.py`: exibe o agente treinado jogando.
- `avaliar_agente.py`: avalia o desempenho sem abrir a janela gráfica.
- `agente_cartpole.keras`: modelo treinado.
- `recompensas_cartpole.csv`: recompensas obtidas durante o treinamento.
- `perdas_cartpole.csv`: perdas da rede neural.
- `grafico_treinamento_cartpole.png`: gráfico da evolução do treinamento.
- `requirements.txt`: dependências do projeto.

## Instalação

Recomenda-se utilizar Python 3.11.

```powershell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
