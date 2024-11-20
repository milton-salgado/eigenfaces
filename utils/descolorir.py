import cv2
import os


def descolorir_arquivo(caminho_imagem):
    """
    Descolore uma imagem, convertendo-a para escala de cinza, e salva no mesmo local.

    Args:
        caminho_imagem (str): Caminho completo do arquivo de imagem a ser descolorido.
    """
    if not os.path.exists(caminho_imagem):
        print(f"Erro: O arquivo '{caminho_imagem}' não foi encontrado.")
        return

    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        print(f"Erro: Não foi possível carregar a imagem '{caminho_imagem}'.")
        return

    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    caminho_saida = os.path.splitext(caminho_imagem)[0] + "_cinza.png"

    cv2.imwrite(caminho_saida, imagem_cinza)
    print(f"Imagem descolorida salva em: {caminho_saida}")


caminho_imagem = '../assets/images/miltin.jpg'
descolorir_arquivo(caminho_imagem)
