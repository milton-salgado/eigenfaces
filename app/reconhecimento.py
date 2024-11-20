import os
import numpy as np
import cv2
from auxiliares import ler_imagens, centralizar_imagem, calcular_autofaces, exibir_imagens


def carregar_bases_de_dados(diretorios):
    """
    Carrega todas as imagens em múltiplos diretórios, incluindo subpastas.

    Args:
        diretorios (list): Lista de caminhos para os diretórios contendo as imagens.

    Returns:
        tuple: Uma lista de imagens normalizadas (float32 entre 0 e 1) e uma lista de rótulos.
    """

    imagens = []
    rotulos = []

    for idx, diretorio in enumerate(diretorios):
        for root, _, files in os.walk(diretorio):
            for file in files:
                if file.endswith((".jpg", ".png")):  # Extensões suportadas
                    caminho_imagem = os.path.join(root, file)
                    imagem = cv2.imread(caminho_imagem)
                    if imagem is not None:
                        imagens.append(np.float32(imagem) / 255.0)
                        rotulos.append(f"Imagem {idx + 1}")

    if not imagens:
        raise ValueError(
            f"Nenhuma imagem válida encontrada nos diretórios: {diretorios}"
        )

    return imagens, rotulos


def calcular_vetores_de_pesos(imagens, face_media, autofaces):
    """
    Calcula o vetor de pesos (projeções nos autovetores) para cada imagem.

    Args:
        imagens (list): Lista de imagens.
        face_media (numpy.array): A face média da base.
        autofaces (list): Lista de autofaces (autovetores).

    Returns:
        numpy.array: Matriz de vetores de pesos, onde cada linha é o vetor de pesos de uma imagem.
    """

    matriz_autofaces = np.array([af.flatten()
                                for af in autofaces])  # Matriz das autofaces
    vetores_de_pesos = []

    for imagem in imagens:
        imagem_centralizada = centralizar_imagem(imagem, face_media)
        pesos = np.dot(matriz_autofaces, imagem_centralizada)
        vetores_de_pesos.append(pesos)

    return np.array(vetores_de_pesos)


def reconhecer_pessoa(imagem_teste, face_media, autofaces, vetores_de_pesos_base, rotulos_base):
    """
    Reconhece a pessoa na imagem de teste, comparando o vetor de pesos com a base.

    Args:
        imagem_teste (numpy.array): Imagem a ser reconhecida.
        face_media (numpy.array): Face média da base.
        autofaces (list): Lista de autofaces (autovetores).
        vetores_de_pesos_base (numpy.array): Matriz de vetores de pesos da base.
        rotulos_base (list): Lista de rótulos correspondentes às imagens da base.

    Returns:
        tuple: Índice do vetor reconhecido, rótulo correspondente e a menor distância.
    """

    imagem_teste_centralizada = centralizar_imagem(imagem_teste, face_media)
    vetor_pesos_teste = np.dot([af.flatten()
                               for af in autofaces], imagem_teste_centralizada)

    distancias = np.linalg.norm(
        vetores_de_pesos_base - vetor_pesos_teste, axis=1)

    indice_reconhecido = np.argmin(distancias)
    return indice_reconhecido, rotulos_base[indice_reconhecido], distancias[indice_reconhecido]


def executar_reconhecimento(diretorio_base, diretorio_teste, num_autofaces=50, limite_base=10000):
    """
    Realiza reconhecimento facial comparando a imagem de teste com uma base de até 10.000 imagens.

    Args:
        diretorio_base (str): Diretório contendo a base de imagens.
        diretorio_teste (str): Diretório contendo a imagem de teste.
        num_autofaces (int): Número de autofaces a serem utilizadas.
        limite_base (int, opcional): Limite máximo de imagens na base. Padrão é 10.000.

    Exibe:
        A imagem de teste e a imagem reconhecida lado a lado usando Matplotlib.
    """

    imagens_base, rotulos_base = carregar_bases_de_dados([diretorio_base])
    if len(imagens_base) > limite_base:
        imagens_base = imagens_base[:limite_base]
        rotulos_base = rotulos_base[:limite_base]
        print(f"Base limitada a {limite_base} imagens para evitar sobrecarga.")

    face_media, autofaces = calcular_autofaces(imagens_base, num_autofaces)

    vetores_de_pesos_base = calcular_vetores_de_pesos(
        imagens_base, face_media, autofaces)

    imagens_teste = ler_imagens(diretorio_teste, limite=1)
    if not imagens_teste:
        raise ValueError(
            "Nenhuma imagem válida encontrada no diretório de teste.")
    imagem_teste = imagens_teste[0]

    indice_reconhecido, rotulo_reconhecido, distancia = reconhecer_pessoa(
        imagem_teste, face_media, autofaces, vetores_de_pesos_base, rotulos_base
    )

    imagem_reconhecida = imagens_base[indice_reconhecido]

    exibir_resultado_reconhecimento(imagem_teste, imagem_reconhecida)


def exibir_resultado_reconhecimento(imagem_teste, imagem_reconhecida):
    """
    Exibe a imagem de teste e a imagem reconhecida lado a lado usando Matplotlib.

    Args:
        imagem_teste (numpy.array): Imagem de teste usada no reconhecimento.
        imagem_reconhecida (numpy.array): Imagem reconhecida a partir da base.
    """

    imagens_para_exibir = [imagem_teste, imagem_reconhecida]
    titulos_imagens = ["Imagem de Teste", "Imagem Reconhecida"]

    exibir_imagens(
        imagens_para_exibir,
        num_colunas=len(imagens_para_exibir),
        titulo_grid="Resultados de Reconhecimento",
        titulos_imagens=titulos_imagens
    )
