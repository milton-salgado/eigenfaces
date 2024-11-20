import os
from PIL import Image


def converter_pgm_para_png(pasta_raiz):
    """
    Converte todas as imagens no formato PGM para PNG em uma pasta e suas subpastas.

    Args:
        pasta_raiz (str): Caminho da pasta raiz onde as imagens estão localizadas.
    """
    for raiz, _, arquivos in os.walk(pasta_raiz):
        for arquivo in arquivos:
            if arquivo.lower().endswith('.pgm'):
                caminho_pgm = os.path.join(raiz, arquivo)

                novo_caminho_png = os.path.join(
                    raiz, os.path.splitext(arquivo)[0] + '.png')

                try:
                    with Image.open(caminho_pgm) as img:
                        img.save(novo_caminho_png, 'PNG')
                        print(f"Convertido: {
                              caminho_pgm} -> {novo_caminho_png}")

                    os.remove(caminho_pgm)
                    print(f"Removido: {caminho_pgm}")

                except Exception as e:
                    print(f"Erro ao converter {caminho_pgm}: {e}")


pasta_raiz = "../assets/images"
converter_pgm_para_png(pasta_raiz)
