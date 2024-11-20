import os
from PIL import Image


def redimensionar_imagens(pasta_raiz, largura=180, altura=220):
    for raiz, _, arquivos in os.walk(pasta_raiz):
        for arquivo in arquivos:
            if arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                caminho_imagem = os.path.join(raiz, arquivo)

                with Image.open(caminho_imagem) as img:
                    img_redimensionada = img.resize((largura, altura))

                    img_redimensionada.save(caminho_imagem)
                    print(f"Imagem redimensionada: {caminho_imagem}")


pastas = ["caminho/para/teste"]

for pasta in pastas:
    redimensionar_imagens(pasta)
