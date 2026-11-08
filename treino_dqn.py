import random
from collections import deque

import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# 1. Configurações do treinamento
# ============================================================

NOME_AMBIENTE = "CartPole-v1"

NUM_EPISODIOS = 300
TAMANHO_LOTE = 64
TAMANHO_MEMORIA = 50_000

TAXA_APRENDIZAGEM = 0.001
FATOR_DESCONTO = 0.99

EPSILON_INICIAL = 1.0
EPSILON_MINIMO = 0.01
DECAY_EPSILON = 0.995

ATUALIZAR_REDE_ALVO_A_CADA = 10

SEMENTE = 42


# ============================================================
# 2. Reprodução dos experimentos
# ============================================================

random.seed(SEMENTE)
np.random.seed(SEMENTE)
tf.random.set_seed(SEMENTE)


# ============================================================
# 3. Memória de experiências
# ============================================================

class ReplayBuffer:
    """
    Armazena experiências vividas pelo agente.

    Cada experiência possui:
    estado, ação, recompensa, próximo_estado e finalizado.
    """

    def __init__(self, capacidade):
        self.memoria = deque(maxlen=capacidade)

    def adicionar(self, estado, acao, recompensa, proximo_estado, finalizado):
        experiencia = (
            estado,
            acao,
            recompensa,
            proximo_estado,
            finalizado
        )

        self.memoria.append(experiencia)

    def amostrar(self, tamanho_lote):
        lote = random.sample(self.memoria, tamanho_lote)

        estados = np.array([experiencia[0] for experiencia in lote])
        acoes = np.array([experiencia[1] for experiencia in lote])
        recompensas = np.array([experiencia[2] for experiencia in lote])
        proximos_estados = np.array([experiencia[3] for experiencia in lote])
        finalizados = np.array([experiencia[4] for experiencia in lote])

        return (
            estados,
            acoes,
            recompensas,
            proximos_estados,
            finalizados
        )

    def __len__(self):
        return len(self.memoria)


# ============================================================
# 4. Criação da rede neural
# ============================================================

def criar_modelo(tamanho_estado, numero_acoes):
    """
    Cria uma rede neural que recebe o estado do jogo
    e retorna o valor estimado de cada ação.
    """
    tamanho_estado = int(tamanho_estado)
    numero_acoes = int(numero_acoes)
    
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(tamanho_estado,)),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),

        # Uma saída para cada ação possível.
        # No CartPole existem duas ações: esquerda e direita.
        tf.keras.layers.Dense(numero_acoes, activation="linear")
    ])

    otimizador = tf.keras.optimizers.Adam(
        learning_rate=TAXA_APRENDIZAGEM
    )

    modelo.compile(
        optimizer=otimizador,
        loss=tf.keras.losses.MeanSquaredError()
    )

    return modelo


# ============================================================
# 5. Escolha da ação
# ============================================================

def escolher_acao(modelo, estado, epsilon, numero_acoes):
    """
    Estratégia epsilon-greedy:

    - Com probabilidade epsilon, escolhe uma ação aleatória;
    - Caso contrário, escolhe a ação com maior valor Q.
    """

    numero_aleatorio = random.random()

    if numero_aleatorio < epsilon:
        return random.randrange(numero_acoes)

    estado = np.asarray(estado, dtype=np.float32)
    estado = np.expand_dims(estado, axis=0)

    valores_q = modelo.predict(estado, verbose=0)

    return int(np.argmax(valores_q[0]))


# ============================================================
# 6. Treinamento da rede neural
# ============================================================

def treinar_modelo(
    modelo,
    rede_alvo,
    memoria,
    tamanho_lote
):
    """
    Treina a rede principal utilizando um lote
    de experiências armazenadas na memória.
    """

    (
        estados,
        acoes,
        recompensas,
        proximos_estados,
        finalizados
    ) = memoria.amostrar(tamanho_lote)

    # Valores Q previstos para os próximos estados
    valores_q_futuros = rede_alvo.predict(
        proximos_estados,
        verbose=0
    )

    maiores_q_futuros = np.max(
        valores_q_futuros,
        axis=1
    )

    # Equação de Bellman:
    #
    # alvo = recompensa + gamma * maior_Q_próximo_estado
    #
    # Se o episódio terminou, não consideramos o próximo estado.
    alvos = recompensas + (
        FATOR_DESCONTO
        * maiores_q_futuros
        * (1 - finalizados.astype(np.float32))
    )

    # Valores Q atuais previstos pela rede principal
    valores_q_atuais = modelo.predict(
        estados,
        verbose=0
    )

    # Atualiza somente o valor Q referente à ação escolhida
    for indice in range(tamanho_lote):
        valores_q_atuais[indice, acoes[indice]] = alvos[indice]

    historico = modelo.fit(
        estados,
        valores_q_atuais,
        epochs=1,
        verbose=0
    )

    return float(historico.history["loss"][0])


# ============================================================
# 7. Programa principal
# ============================================================

def main():
    print("Criando o ambiente...")

    env = gym.make(NOME_AMBIENTE)

    tamanho_estado = int(env.observation_space.shape[0])
    numero_acoes = int(env.action_space.n)

    print(f"Tamanho do estado: {tamanho_estado}")
    print(f"Número de ações: {numero_acoes}")

    modelo = criar_modelo(
        tamanho_estado,
        numero_acoes
    )

    rede_alvo = criar_modelo(
        tamanho_estado,
        numero_acoes
    )

    # Inicialmente, as duas redes possuem os mesmos pesos
    rede_alvo.set_weights(modelo.get_weights())

    memoria = ReplayBuffer(TAMANHO_MEMORIA)

    epsilon = EPSILON_INICIAL

    recompensas_por_episodio = []
    perdas_por_episodio = []

    print("\nIniciando treinamento...\n")

    for episodio in range(1, NUM_EPISODIOS + 1):

        estado, info = env.reset()

        recompensa_total = 0
        perdas_do_episodio = []

        while True:

            acao = escolher_acao(
                modelo,
                estado,
                epsilon,
                numero_acoes
            )

            (
                proximo_estado,
                recompensa,
                terminated,
                truncated,
                info
            ) = env.step(acao)

            finalizado = terminated or truncated

            memoria.adicionar(
                estado,
                acao,
                recompensa,
                proximo_estado,
                finalizado
            )

            estado = proximo_estado
            recompensa_total += recompensa

            # O treinamento começa quando já existem
            # experiências suficientes na memória.
            if len(memoria) >= TAMANHO_LOTE:
                perda = treinar_modelo(
                    modelo,
                    rede_alvo,
                    memoria,
                    TAMANHO_LOTE
                )

                perdas_do_episodio.append(perda)

            if finalizado:
                break

        # Reduz gradualmente a exploração aleatória
        if epsilon > EPSILON_MINIMO:
            epsilon *= DECAY_EPSILON
            epsilon = max(epsilon, EPSILON_MINIMO)

        # Atualiza a rede-alvo periodicamente
        if episodio % ATUALIZAR_REDE_ALVO_A_CADA == 0:
            rede_alvo.set_weights(modelo.get_weights())

        recompensas_por_episodio.append(recompensa_total)

        if perdas_do_episodio:
            perda_media = np.mean(perdas_do_episodio)
        else:
            perda_media = 0.0

        perdas_por_episodio.append(perda_media)

        if episodio == 1 or episodio % 10 == 0:
            ultimas_recompensas = recompensas_por_episodio[-10:]
            media_ultimas = np.mean(ultimas_recompensas)

            print(
                f"Episódio {episodio:03d}/{NUM_EPISODIOS} | "
                f"Recompensa: {recompensa_total:6.1f} | "
                f"Média últimos 10: {media_ultimas:6.2f} | "
                f"Epsilon: {epsilon:.4f} | "
                f"Perda: {perda_media:.5f}"
            )

    env.close()

    # ========================================================
    # 8. Salvamento do modelo e dos resultados
    # ========================================================

    modelo.save("agente_cartpole.keras")

    np.savetxt(
        "recompensas_cartpole.csv",
        np.array(recompensas_por_episodio),
        delimiter=",",
        header="recompensa",
        comments=""
    )

    np.savetxt(
        "perdas_cartpole.csv",
        np.array(perdas_por_episodio),
        delimiter=",",
        header="perda",
        comments=""
    )

    # ========================================================
    # 9. Geração do gráfico
    # ========================================================

    media_movel = []

    janela = 20

    for indice in range(len(recompensas_por_episodio)):
        inicio = max(0, indice - janela + 1)
        media = np.mean(recompensas_por_episodio[inicio:indice + 1])
        media_movel.append(media)

    plt.figure(figsize=(10, 5))

    plt.plot(
        recompensas_por_episodio,
        alpha=0.4,
        label="Recompensa por episódio"
    )

    plt.plot(
        media_movel,
        linewidth=2,
        label="Média móvel de 20 episódios"
    )

    plt.xlabel("Episódio")
    plt.ylabel("Recompensa")
    plt.title("Treinamento do agente DQN no CartPole-v1")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        "grafico_treinamento_cartpole.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print("\nTreinamento concluído.")
    print("Modelo salvo como: agente_cartpole.keras")
    print("Resultados salvos em: recompensas_cartpole.csv")
    print("Gráfico salvo como: grafico_treinamento_cartpole.png")


if __name__ == "__main__":
    main()
