import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from glob import glob


def ler_imagens(diretorio, limite=None):
    """
    Lê todas as imagens de um diretório e seus subdiretórios, até o limite especificado.
    Caso o diretório contenha apenas uma imagem, retorna essa única imagem como elemento
    dentro de uma lista.

    Args:
        diretorio (str): Caminho do diretório contendo as imagens.
        limite (int, opcional): Número máximo de imagens a carregar. Padrão é None.

    Returns:
        list: Lista de imagens carregadas como arrays numpy normalizados.
    """

    diretorio = os.path.normpath(diretorio)
    imagens = []

    if os.path.isfile(diretorio):
        imagem = cv2.imread(diretorio)
        if imagem is None:
            raise ValueError(
                f"O arquivo especificado não é uma imagem válida: {diretorio}")
        return [np.float32(imagem) / 255.0]

    formatos = ("*.jpg", "*.png")
    arquivos = []
    for formato in formatos:
        arquivos.extend(
            glob(os.path.join(diretorio, "**", formato), recursive=True))

    if len(arquivos) == 1:
        imagem = cv2.imread(arquivos[0])
        if imagem is None:
            raise ValueError(
                f"O arquivo encontrado não é uma imagem válida: {arquivos[0]}")
        return [np.float32(imagem) / 255.0]

    for i, arquivo_imagem in enumerate(arquivos):
        if limite is not None and i >= limite:
            break
        imagem = cv2.imread(arquivo_imagem)
        if imagem is not None:
            imagens.append(np.float32(imagem) / 255.0)

    if not imagens:
        raise ValueError(
            f"Nenhuma imagem válida encontrada no diretório: {diretorio}")

    return imagens


def centralizar_imagem(imagem, face_media):
    """
    Centraliza uma imagem subtraindo a face média.

    Args:
        imagem (numpy.array): Imagem a ser centralizada.
        face_media (numpy.array): Face média da base.

    Returns:
        numpy.array: Imagem centralizada.
    """
    return imagem.flatten() - face_media.flatten()


def calcular_autofaces(imagens, num_autofaces=15):
    """
    Calcula a face média e as autofaces (autovetores) de forma otimizada,
    utilizando a matriz de covariância reduzida.

    Args:
        imagens (list): Lista de imagens da base, cada uma representada como um array numpy.
        num_autofaces (int, opcional): Número de autofaces a serem calculadas. Padrão é 15.

    Returns:
        tuple: Face média (numpy.array) e lista de autofaces (list).
    """

    dados = np.array([imagem.flatten()
                     for imagem in imagens], dtype=np.float32)

    face_media = np.mean(dados, axis=0)

    dados_centralizados = dados - face_media

    matriz_reduzida = np.dot(dados_centralizados, dados_centralizados.T)

    autovalores, autovetores_reduzidos = np.linalg.eigh(matriz_reduzida)

    indices = np.argsort(autovalores)[::-1]
    autovalores = autovalores[indices]
    autovetores_reduzidos = autovetores_reduzidos[:, indices]

    autovetores_reduzidos = autovetores_reduzidos[:, :num_autofaces]

    autovetores_originais = np.dot(
        dados_centralizados.T, autovetores_reduzidos)

    autovetores_norm = np.array([
        vetor / np.linalg.norm(vetor) for vetor in autovetores_originais.T
    ])

    autofaces = [vetor.reshape(imagens[0].shape) for vetor in autovetores_norm]

    return face_media.reshape(imagens[0].shape), autofaces


def exibir_imagens(imagens, num_colunas=5, titulo_grid="Grid de Imagens", titulos_imagens=None):
    """
    Exibe um grid de imagens utilizando Matplotlib.

    Args:
        imagens (list): Lista de arrays numpy representando as imagens a serem exibidas.
        num_colunas (int, opcional): Número de colunas no grid. Padrão é 5.
        titulo_grid (str, opcional): Título geral do grid. Padrão é "Grid de Imagens".
        titulos_imagens (list, opcional): Lista de títulos para cada imagem. Padrão é None.
        cmap (str, opcional): Mapa de cores a ser utilizado. Padrão é "gray".
    """
    num_linhas = -(-len(imagens) // num_colunas)

    fig, axes = plt.subplots(num_linhas, num_colunas,
                             figsize=(4 * num_colunas, 4 * num_linhas))

    if num_linhas == 1 and num_colunas == 1:
        axes = [axes]
    elif num_linhas == 1 or num_colunas == 1:
        axes = np.array(axes).flatten()
    else:
        axes = axes.flatten()

    for i, imagem in enumerate(imagens):
        axes[i].imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
        if titulos_imagens:
            axes[i].set_title(titulos_imagens[i], fontsize=10)
        axes[i].axis("off")

    for j in range(len(imagens), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(titulo_grid, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def exibir_autofaces(autofaces, titulo="Autofaces"):
    """
    Exibe as autofaces em um grid utilizando Matplotlib.

    Args:
        autofaces (list): Lista de arrays numpy representando as autofaces.
    """

    autofaces_normalizadas = normalizar_autofaces(autofaces)
    titulos = [f"Autoface {i + 1}" for i in range(len(autofaces))]
    exibir_imagens(autofaces_normalizadas, num_colunas=5,
                   titulo_grid=titulo, titulos_imagens=titulos)


def exibir_faces_medias(faces_medias):
    """
    Exibe a face média em um grid utilizando Matplotlib.

    Args:
        face_media (numpy.array): Array numpy representando a face média.
    """

    titulo = "Faces Médias"
    titulos_imagens = [f"Face Média {i + 1}" for i in range(len(faces_medias))]
    exibir_imagens(faces_medias, num_colunas=3,
                   titulo_grid=titulo, titulos_imagens=titulos_imagens)


def normalizar_autofaces(autofaces):
    """
    Normaliza as autofaces para o intervalo [0, 1] para exibição.

    Args:
        autofaces (list): Lista de arrays numpy representando as autofaces.

    Returns:
        list: Lista de autofaces normalizadas.
    """

    return [(
        af - np.min(af)) / (np.max(af) - np.min(af)) for af in autofaces]
