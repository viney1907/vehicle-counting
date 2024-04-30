# Importa as bibliotecas necessárias
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

# Carrega o modelo YOLO com um modelo pré-treinado especificado
model = YOLO("yolov8n.pt")

# Abre o arquivo de vídeo localizado no caminho especificado
cap = cv2.VideoCapture("C:/Users/User/Documents/Programação/vehicle-counting/vehicle-counting/data/videos/video.mp4")
# Verifica se o vídeo foi aberto corretamente
assert cap.isOpened(), "Error reading video file"

# Obtém largura, altura e taxa de quadros por segundo (FPS) do vídeo
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define os pontos de uma região de interesse no vídeo para contagem de objetos
line_points = [(170, 400), (1100, 400)]  # line or region points
classes_to_count = [0, 2]  # person and car classes for count

# Configura o escritor de vídeo para gravar o vídeo processado com o codec MP4V
video_writer = cv2.VideoWriter("object_counting_output.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

# Inicializa o contador de objetos
counter = object_counter.ObjectCounter()
# Configura o contador de objetos com várias opções
counter.set_args(view_img=True,           # Permite visualização das imagens processadas
                 reg_pts=line_points,   # Define os pontos da região de interesse
                 classes_names=model.names, # Usa nomes de classes do modelo YOLO
                 draw_tracks=True,        # Habilita o desenho das trajetórias dos objetos
                 line_thickness=2)        # Define a espessura das linhas para desenhar trajetórias

# Processa o vídeo quadro a quadro
while cap.isOpened():
    # Lê o próximo quadro do vídeo
    success, im0 = cap.read()
    # Verifica se o quadro foi lido corretamente
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    # Realiza a detecção e o rastreamento de objetos no quadro
    tracks = model.track(im0, persist=True, show=False)
    # Inicia a contagem de objetos e atualiza o quadro com informações visuais
    im0 = counter.start_counting(im0, tracks)
    # Escreve o quadro processado no arquivo de saída
    video_writer.write(im0)

# Libera os recursos do vídeo e do escritor de vídeo
cap.release()
video_writer.release()
# Fecha todas as janelas abertas pelo OpenCV
cv2.destroyAllWindows()
