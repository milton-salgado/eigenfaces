from matplotlib import pyplot as plt
import numpy as np
from auxiliares import ler_imagens, calcular_autofaces, exibir_imagens, centralizar_imagem


def projetar_imagens(imagens, face_media, autofaces):
    """
    Projeta as imagens no espaço gerado pelas autofaces.

    Args:
        imagens (list): Lista de imagens a serem projetadas.
        face_media (numpy.array): Face média da base.
        autofaces (list): Lista de autofaces (autovetores).

    Returns:
        numpy.array: Matriz de projeções no espaço das autofaces.
    """

    matriz_autofaces = np.array([af.flatten() for af in autofaces])

    projecoes = []
    for imagem in imagens:
        imagem_centralizada = centralizar_imagem(imagem, face_media)
        alfa = np.dot(matriz_autofaces, imagem_centralizada)
        projecoes.append(alfa)

    return np.array(projecoes)


def plotar_projecoes(projecoes, rotulos):
    """
    Plota as projeções no espaço tridimensional usando matplotlib.

    Args:
        projecoes (numpy.array): Matriz de projeções (pesos).
        rotulos (list): Lista de rótulos correspondentes às imagens projetadas.
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    cores = ['red', 'blue', 'green']
    marcadores = ['o', '^', 's']

    for i, rotulo in enumerate(np.unique(rotulos)):
        indices = [j for j, r in enumerate(rotulos) if r == rotulo]
        ax.scatter(
            projecoes[indices, 0],
            projecoes[indices, 1],
            projecoes[indices, 2],
            label=f"Classe {rotulo}",
            color=cores[i % len(cores)],
            marker=marcadores[i % len(marcadores)],
            s=50
        )

    ax.set_xlabel("Alpha 1 (Autoface 1)")
    ax.set_ylabel("Alpha 2 (Autoface 2)")
    ax.set_zlabel("Alpha 3 (Autoface 3)")
    ax.set_title("Projeções no Subespaço 3D")
    ax.legend()

    plt.show()


def exibir_imagens_pessoas(diretorios):
    """
    Exibe um grid de imagens para cada pessoa com 10 imagens em 2 linhas e 5 colunas.

    Args:
        diretorios (list): Lista de caminhos para os diretórios contendo as imagens de cada pessoa.
    """

    for i, diretorio in enumerate(diretorios):

        imagens = ler_imagens(diretorio, limite=10)
        titulos = [f"Imagem {j + 1}" for j in range(len(imagens))]

        exibir_imagens(imagens, num_colunas=5, titulo_grid=f"Pessoa {
                       i + 1}", titulos_imagens=titulos)


def executar_classificacao(diretorios, num_autofaces=3):
    """
    Executa o processo de classificação das imagens no espaço das autofaces para múltiplos diretórios.

    Args:
        diretorios (list): Lista de caminhos para os diretórios contendo as imagens de cada pessoa.
        num_autofaces (int, opcional): Número de autofaces utilizadas. Padrão é 3.

    Exibe:
        Um gráfico 3D das projeções no espaço das autofaces, com cores diferentes para cada pessoa.
    """

    exibir_imagens_pessoas(diretorios)

    todas_imagens = []
    rotulos = []

    for i, diretorio in enumerate(diretorios):
        imagens = ler_imagens(diretorio, limite=10)
        todas_imagens.extend(imagens)
        rotulos.extend([f"Pessoa {i + 1}"] * len(imagens))

    face_media, autofaces = calcular_autofaces(todas_imagens, num_autofaces)

    projecoes = projetar_imagens(todas_imagens, face_media, autofaces)

    plotar_projecoes(projecoes, rotulos)
