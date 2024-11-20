import numpy as np
from auxiliares import ler_imagens, calcular_autofaces, centralizar_imagem, exibir_imagens


def aproximar_imagem(imagem, face_media, autofaces):
    """
    Aproxima uma imagem da base usando a combinação linear das autofaces.

    Args:
        imagem (numpy.array): Imagem original que será aproximada.
        face_media (numpy.array): Face média calculada para a base.
        autofaces (list): Lista de autofaces calculadas.

    Returns:
        numpy.array: Imagem aproximada reconstruída.
    """

    matriz_autofaces = np.array([af.flatten() for af in autofaces])

    imagem_centralizada = centralizar_imagem(imagem, face_media)

    pesos = np.dot(matriz_autofaces, imagem_centralizada)

    imagem_aproximada = np.dot(
        matriz_autofaces.T, pesos) + face_media.flatten()

    return imagem_aproximada.reshape(face_media.shape)


def executar_aproximacao(diretorio_imagens, imagem_teste, lista_autofaces, limite=None):
    """
    Aproxima uma imagem utilizando diferentes números de autofaces e exibe os resultados no Matplotlib.

    Args:
        diretorio_imagens (str): Caminho do diretório contendo as imagens da base.
        imagem_teste (numpy.array): Imagem a ser aproximada.
        lista_autofaces (list): Lista com os números de autofaces a serem utilizados.
        limite (int): Limite máximo de imagens a serem carregadas. Padrão é None.

    Exibe:
        Um grid com a imagem original e as aproximações geradas.
    """

    imagens = ler_imagens(diretorio_imagens, limite=limite)

    face_media, _ = calcular_autofaces(imagens, max(lista_autofaces))

    imagens_para_exibir = [imagem_teste]
    titulos_imagens = ["Imagem Original"]

    for num_autofaces in lista_autofaces:
        _, autofaces = calcular_autofaces(imagens, num_autofaces)
        imagem_aproximada = aproximar_imagem(
            imagem_teste, face_media, autofaces)
        imagens_para_exibir.append(imagem_aproximada)
        titulos_imagens.append(f"{num_autofaces} Autofaces")

    exibir_imagens(
        imagens_para_exibir,
        num_colunas=min(4, len(imagens_para_exibir)),
        titulo_grid="Resultados de Aproximação",
        titulos_imagens=titulos_imagens
    )
