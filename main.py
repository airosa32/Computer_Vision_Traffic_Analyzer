import os
import time
from typing import Tuple
import cv2
from tracker import LineCounter
from ultralytics import YOLO


def process_video(
    input_path: str, output_path: str, model_weights: str = 'yolov8n.pt'
) -> None:
  """Executa a pipeline de inferência, rastreamento (ByteTrack) e contagem de pedestres em vídeo.

  Args:
      input_path (str): Caminho do vídeo de entrada.
      output_path (str): Caminho onde o vídeo processado será salvo.
      model_weights (str): Pesos do modelo YOLOv8 a serem utilizados.
  """
  # Garante que o diretório de saída exista
  os.makedirs(os.path.dirname(output_path), exist_ok=True)

  # Carrega o modelo de detecção de objetos
  model = YOLO(model_weights)

  # Abre o arquivo de vídeo
  cap = cv2.VideoCapture(input_path)
  if not cap.isOpened():
    print(f'[ERRO] Não foi possível abrir o vídeo: {input_path}')
    return

  # Obtém propriedades do vídeo de entrada
  width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps: int = int(cap.get(cv2.CAP_PROP_FPS)) or 30

  # Configura o gravador do vídeo anotado
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

  # Configura a Linha Virtual (ROI) na região central do vídeo
  line_p1: Tuple[int, int] = (50, int(height * 0.55))
  line_p2: Tuple[int, int] = (width - 50, int(height * 0.55))
  counter = LineCounter(line_p1, line_p2)

  prev_time = time.time()
  print('Iniciando processamento e rastreamento de vídeo...\n')

  try:
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break

      # Executa inferência com rastreamento persistente (ByteTrack)
      results = model.track(
          frame, persist=True, tracker='bytetrack.yaml', verbose=False
      )

      tracks = []
      if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        for bbox, track_id, cls_id in zip(boxes, track_ids, clss):
          # Filtra estritamente a classe 0 (Person no dataset COCO)
          if cls_id == 0:
            tracks.append((track_id, bbox, cls_id))

            # Desenha Bounding Box e ID Persistente
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 105, 180), 2)
            cv2.putText(
                frame,
                f'ID: {track_id}',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 105, 180),
                2,
            )

      # Atualiza a lógica de cruzamento e renderiza a linha
      counter.update(tracks)
      counter.draw(frame)

      # Cálculo de FPS em tempo real
      curr_time = time.time()
      current_fps = 1.0 / (curr_time - prev_time + 1e-6)
      prev_time = curr_time

      # Dashboard HUD no topo do frame
      cv2.rectangle(frame, (20, 20), (240, 125), (0, 0, 0), -1)
      cv2.putText(
          frame,
          f'IN  : {counter.in_count}',
          (30, 50),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (0, 255, 0),
          2,
      )
      cv2.putText(
          frame,
          f'OUT : {counter.out_count}',
          (30, 85),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (0, 0, 255),
          2,
      )
      cv2.putText(
          frame,
          f'FPS : {int(current_fps)}',
          (30, 110),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (255, 255, 255),
          1,
      )

      # Grava o frame anotado no arquivo de saída
      out.write(frame)

  finally:
    # Libera os recursos de mídia
    cap.release()
    out.release()

  # Relatório final impresso no terminal
  total_processed = counter.in_count + counter.out_count

  print('\n' + '=' * 45)
  print('       RELATÓRIO DE CONTAGEM DE TRÁFEGO')
  print('=' * 45)
  print(f' Entradas (IN)   : {counter.in_count}')
  print(f' Saídas (OUT)    : {counter.out_count}')
  print(f' Total Medido    : {total_processed} pedestres')
  print('=' * 45)
  print(f' Vídeo processado salvo em:')
  print(f' -> {output_path}')
  print('=' * 45 + '\n')


if __name__ == '__main__':
  INPUT_VIDEO = 'videos/sample.mp4'
  OUTPUT_VIDEO = 'output/result_video.mp4'

  process_video(INPUT_VIDEO, OUTPUT_VIDEO)