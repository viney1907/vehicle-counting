import cv2
import torch

# Função para carregar o modelo pré-treinado YOLOv5
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

# Função para processar o vídeo
def process_video(video_path, model):
    # Abrir o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Preparar a gravação do vídeo de saída (opcional)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('outputs/processed_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Convert frame to the format YOLO expects
            results = model(frame)

            # Desenhar caixas e rótulos nos veículos detectados
            results.render()

            # Mostrar o quadro processado
            cv2.imshow('Vehicle Detection', frame)

            # Gravar o quadro processado no vídeo de saída (opcional)
            out.write(frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Carregar o modelo
model = load_model()

# Caminho do vídeo de entrada
video_path = 'data/videos/your_video.mp4'

# Processar o vídeo
process_video(video_path, model)
