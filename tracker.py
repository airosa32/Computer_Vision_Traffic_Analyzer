from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np


class LineCounter:
  """Classe para contagem de objetos que cruzam uma linha virtual (ROI).

  Utiliza produto vetorial 2D (determinante) para identificar a mudança de lado
  e a direção do movimento dos objetos rastreados (IN / OUT).
  """

  def __init__(
      self,
      line_point1: Tuple[int, int],
      line_point2: Tuple[int, int],
  ) -> None:
    """Inicializa o contador da linha virtual.

    Args:
        line_point1 (Tuple[int, int]): Coordenada (x, y) do ponto inicial da
          linha.
        line_point2 (Tuple[int, int]): Coordenada (x, y) do ponto final da
          linha.
    """
    self.p1: np.ndarray = np.array(line_point1, dtype=np.int32)
    self.p2: np.ndarray = np.array(line_point2, dtype=np.int32)

    # Estado dos rastreamentos ativos: {track_id: (cx, cy)}
    self.tracker_state: Dict[int, Tuple[int, int]] = {}

    # Métricas de contagem
    self.in_count: int = 0
    self.out_count: int = 0

    # Conjunto de IDs que já cruzaram a linha (evita duplicidade de contagem)
    self.crossed_ids: Set[int] = set()

  def _is_crossed(
      self, prev_pos: Tuple[int, int], curr_pos: Tuple[int, int]
  ) -> Optional[str]:
    """Calcula a orientação relativa do objeto em relação à linha virtual

    utilizando o produto vetorial 2D: (ax * by - ay * bx).

    Nota de Compatibilidade: Abordagem escalar 2D totalmente imune a quebras
    de API entre versões do NumPy (1.x e 2.x+).

    Args:
        prev_pos (Tuple[int, int]): Posição anterior do centroide (x, y).
        curr_pos (Tuple[int, int]): Posição atual do centroide (x, y).

    Returns:
        Optional[str]: 'IN' para transição em uma direção, 'OUT' para direção
        oposta,
                       ou None caso não haja cruzamento no frame atual.
    """
    # Vetor diretor da linha virtual
    v_line = self.p2 - self.p1

    # Vetores do ponto inicial da linha até o histórico de posições do objeto
    v_prev = np.array(prev_pos, dtype=np.int32) - self.p1
    v_curr = np.array(curr_pos, dtype=np.int32) - self.p1

    # Produto vetorial 2D (Determinante): o sinal indica o lado da linha (+ ou -)
    cross_prev = v_line[0] * v_prev[1] - v_line[1] * v_prev[0]
    cross_curr = v_line[0] * v_curr[1] - v_line[1] * v_curr[0]

    # Se o produto dos sinais for negativo, o objeto trocou de lado da linha
    if (cross_prev * cross_curr) < 0:
      # Transição do lado negativo para o positivo
      if cross_prev < 0 and cross_curr > 0:
        return 'IN'
      # Transição do lado positivo para o negativo
      elif cross_prev > 0 and cross_curr < 0:
        return 'OUT'

    return None

  def update(self, tracks: List[Tuple[int, np.ndarray, int]]) -> None:
    """Processa a lista de objetos rastreados no frame atual e atualiza contadores.

    Args:
        tracks (List[Tuple[int, np.ndarray, int]]): Lista contendo
          tuplas no formato (track_id, bounding_box [x1, y1, x2, y2], class_id).
    """
    for track_id, bbox, _ in tracks:
      # Cálculo do ponto central (centroide) do bounding box
      cx = int((bbox[0] + bbox[2]) / 2)
      cy = int((bbox[1] + bbox[3]) / 2)
      curr_pos = (cx, cy)

      # Processa histórico de movimento caso o objeto já tenha sido registrado
      if track_id in self.tracker_state:
        prev_pos = self.tracker_state[track_id]

        # Valida cruzamento apenas se o ID ainda não foi contabilizado
        if track_id not in self.crossed_ids:
          direction = self._is_crossed(prev_pos, curr_pos)

          if direction == 'IN':
            self.in_count += 1
            self.crossed_ids.add(track_id)
          elif direction == 'OUT':
            self.out_count += 1
            self.crossed_ids.add(track_id)

      # Atualiza a posição atual do ID para análise do próximo frame
      self.tracker_state[track_id] = curr_pos

  def draw(self, frame: np.ndarray) -> np.ndarray:
    """Renderiza a linha virtual de contagem no frame de vídeo.

    Args:
        frame (np.ndarray): Frame de vídeo em formato OpenCV (BGR).

    Returns:
        np.ndarray: Frame anotado com a linha virtual renderizada.
    """
    # Desenha a linha amarela (BGR: 0, 255, 255) com espessura de 3px
    cv2.line(frame, tuple(self.p1), tuple(self.p2), (0, 255, 255), 3)
    return frame