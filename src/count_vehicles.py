import cv2
import torch


# Função para carregar o modelo pré-treinado YOLOv5
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.4
    model.iou = 0.5
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
    out = cv2.VideoWriter('C:/Users/vinic/Documents/Vehicle_Counter/vehicle-counting/outputs/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    # Definir a linha de contagem
    line_position = 550
    vehicle_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Convert frame to the format YOLO expects
            results = model(frame)
            # Filtrar detecções de veículos
            results = post_process_detections(results, frame_width, frame_height)

            # Desenhar a linha de contagem
            cv2.line(frame, (0, line_position), (frame_width, line_position), (255, 0, 0), 3)

            current_frame_vehicles = []

            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls_id = det.tolist()
                center_y = (y1 + y2) / 2

                # Verificar se o centro do veiculo cruzou a linha de contagem
                if center_y < line_position + 2 and center_y > line_position - 2:
                    vehicle_count +=1


            # Desenhar caixas e rótulos nos veículos detectados
            results.render()

            # Texto com a contagem de veículos
            cv2.putText(frame, f"Vehicle count: {vehicle_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
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

    print(f"Total vehicles counted: {vehicle_count}")


def post_process_detections(results, frame_width, frame_height):
    # Filtro de tamanho para caixas delimitadoras
    min_area = 1500  # Mínimo de área para detecções
    max_area = 30000  # Máximo de área para detecções
    valid_detections = []

    detections = results.xyxy[0]  # Tensor de detecções
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        area = (x2 - x1) * (y2 - y1)
        if min_area < area < max_area:
            valid_detections.append(det.tolist())

    # Convertendo a lista filtrada de volta para um tensor
    if valid_detections:
        results.xyxy[0] = torch.tensor(valid_detections, device=detections.device)
    else:
        # Se nenhum filtro passar, criamos um tensor vazio com a mesma estrutura
        results.xyxy[0] = torch.empty((0, 6), device=detections.device)

    return results

# Carregar o modelo
model = load_model()

# Caminho do vídeo de entrada
video_path = 'C:/Users/vinic/Documents/Vehicle_Counter/vehicle-counting/data/videos/video.mp4'

# Processar o vídeo
process_video(video_path, model)
