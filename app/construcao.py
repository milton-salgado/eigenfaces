import cv2
from auxiliares import ler_imagens, calcular_autofaces


def executar_construcao(diretorio_imagens, num_autofaces=15, limite=400):
    """
    Exibe a interface para manipulação interativa das autofaces e construção de novas imagens.

    Args:
        diretorio_imagens (str): Caminho do diretório contendo as imagens da base.
        num_autofaces (int, opcional): Número de autofaces a serem utilizadas. Padrão é 15.
        limite (int, opcional): Limite máximo de imagens a serem carregadas. Padrão é 400.

    Funcionalidade:
        - Calcula a face média e as autofaces da base de dados.
        - Cria sliders para ajustar os pesos das autofaces.
        - Atualiza e exibe uma nova imagem reconstruída com base nos pesos ajustados pelos sliders.
        - A interface usa OpenCV para exibir duas janelas:
            1. "Resultado": Mostra a imagem reconstruída.
            2. "Autofaces e Pesos": Contém os sliders para ajustar os pesos das autofaces.
    """

    imagens = ler_imagens(diretorio_imagens, limite=limite)
    face_media, autofaces = calcular_autofaces(imagens, num_autofaces)

    cv2.namedWindow("Resultado", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("Autofaces e Pesos", cv2.WINDOW_GUI_NORMAL)

    valores_slider = [127] * num_autofaces

    def atualizar_nova_face(*args):
        """
        Atualiza a imagem reconstruída com base nos pesos ajustados pelos sliders.
        """
        nova_face = face_media.copy()
        for i, autoface in enumerate(autofaces):
            peso = cv2.getTrackbarPos(f"Peso {i}", "Autofaces e Pesos") - 127
            nova_face += autoface * peso
        # Atualiza a janela de resultados
        resultado_atualizado = cv2.resize(nova_face, (0, 0), fx=2, fy=2)
        cv2.imshow("Resultado", resultado_atualizado)

    for i in range(num_autofaces):
        cv2.createTrackbar(
            f"Peso {i}", "Autofaces e Pesos", 127, 255, atualizar_nova_face)

    while cv2.getWindowProperty("Resultado", cv2.WND_PROP_VISIBLE) >= 1 and \
            cv2.getWindowProperty("Autofaces e Pesos", cv2.WND_PROP_VISIBLE) >= 1:
        cv2.waitKey(1)

    cv2.destroyAllWindows()
