# Computer Vision Traffic Analyzer

Application developed in Python for real-time video processing, persistent object tracking, and bi-directional pedestrian line counting using YOLOv8, ByteTrack, and OpenCV.

This project is part of a progressive engineering portfolio series, demonstrating architectural evolution from local computer vision systems to distributed architectures and MLOps pipelines.

---

## Features

* **Persistent Object Tracking:** Integrates ByteTrack with YOLOv8 to assign and maintain unique persistent IDs for pedestrians across frames.
* **Bi-Directional ROI Line Counting:** Employs 2D vector mathematics (cross-product orientation) to detect precise crossings and classify movement directions (`IN` / `OUT`).
* **Real-Time HUD Dashboard:** Displays real-time FPS metrics, individual persistent IDs, and current traffic counts rendered on top of video frames.
* **Robust Vector Math:** Uses scalar determinant calculations compatible across NumPy 1.x and 2.x+ environments without numeric API breakage.
* **Modular Architecture:** Decouples vector line-crossing logic (`tracker.py`) from video stream ingestion and reporting (`main.py`).

---

## Project Structure

```text
computer-vision-traffic-analyzer/
├── videos/             # Input video files
├── output/             # Processed video output (result_video.mp4)
├── tracker.py          # LineCounter class (2D Vector Cross Product logic)
├── main.py             # Video execution pipeline & CLI reporting
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

---

## Results & Demonstration

### Video Demo

The following demonstration shows the complete detection, tracking, and bi-directional pedestrian counting pipeline.

![Traffic Analyzer Demo](output/result_video.gif)

> **Output:** `output/result_video.gif`

The processed video includes:

* YOLOv8 pedestrian detection
* ByteTrack persistent tracking IDs
* Real-time FPS information
* `IN` / `OUT` traffic classification
* Live pedestrian count dashboard
* Line-crossing visualization

### Terminal Output

```text
Iniciando processamento e rastreamento de vídeo...

=============================================
       RELATÓRIO DE CONTAGEM DE TRÁFEGO
=============================================
 Entradas (IN)   : 8
 Saídas (OUT)    : 5
 Total Medido    : 13 pedestres
=============================================
 Vídeo processado salvo em:
 -> output/result_video.mp4
=============================================
```

### Traffic Counting Result

| Direction | Pedestrians |
| :-------: | ----------: |
|   **IN**  |           8 |
|  **OUT**  |           5 |
| **Total** |      **13** |

The final report provides a summary of the pedestrians detected crossing the defined ROI line in each direction.
