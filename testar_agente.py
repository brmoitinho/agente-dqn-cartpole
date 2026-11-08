import time
import numpy as np
import gymnasium as gym
import tensorflow as tf


NOME_AMBIENTE = "CartPole-v1"
CAMINHO_MODELO = "modelos/agente_cartpole.keras"
NUMERO_DE_EPISODIOS = 5


def escolher_melhor_acao(modelo, estado):
    """
    Escolhe a ação com maior valor Q previsto pela rede.
    Durante o teste não usamos ações aleatórias.
    """

    estado = np.asarray(estado, dtype=np.float32)
    estado = np.expand_dims(estado, axis=0)

    valores_q = modelo.predict(estado, verbose=0)

    return int(np.argmax(valores_q[0]))


def main():
    print("Carregando o modelo treinado...")

    modelo = tf.keras.models.load_model(CAMINHO_MODELO)

    # render_mode="human" abre uma janela com a animação do jogo
    env = gym.make(
        NOME_AMBIENTE,
        render_mode="human"
    )

    recompensas = []

    for episodio in range(1, NUMERO_DE_EPISODIOS + 1):
        estado, info = env.reset()

        recompensa_total = 0
        passo = 0

        while True:
            acao = escolher_melhor_acao(modelo, estado)

            estado, recompensa, terminated, truncated, info = env.step(acao)

            recompensa_total += recompensa
            passo += 1

            if terminated or truncated:
                break

        recompensas.append(recompensa_total)

        print(
            f"Episódio {episodio}: "
            f"{passo} passos | "
            f"Recompensa total: {recompensa_total}"
        )

        # Aguarda um pouco antes de iniciar o próximo episódio
        time.sleep(1)

    env.close()

    media = np.mean(recompensas)

    print("\nTeste concluído.")
    print(f"Recompensas: {recompensas}")
    print(f"Recompensa média: {media:.2f}")


if __name__ == "__main__":
    main()
