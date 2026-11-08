import gymnasium as gym

# Cria o ambiente CartPole
env = gym.make("CartPole-v1")

# Inicializa o ambiente
observation, info = env.reset(seed=42)

print("Estado inicial:", observation)
print("Espaço de observações:", env.observation_space)
print("Espaço de ações:", env.action_space)

# Executa dez passos usando ações aleatórias
for passo in range(10):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    print(
        f"Passo: {passo + 1}, "
        f"Ação: {action}, "
        f"Recompensa: {reward}, "
        f"Estado: {observation}"
    )

    # O episódio pode terminar por sucesso ou por limite de tempo
    if terminated or truncated:
        print("O episódio terminou. Reiniciando...")
        observation, info = env.reset()

env.close()
