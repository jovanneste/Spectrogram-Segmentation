## Reseach into spectrogram segmentation for underwater noise classification 

### YOLO
We use transfer learning on yolov7 to detect bounding boxes around potential dolphin whistles.


| Data Type        | # Training Labels | Precision | Recall | mAP@0.5 |
|------------------|-------------------|-----------|--------|---------|
| REP23            | 5000              | 0.85      | 0.80   | 0.82    |
| REP23+21         | 7000              | 0.78      | 0.75   | 0.77    |
| REP23+21+aug     | 10000             | 0.90      | 0.88   | 0.89    |
| REP23+21+aug+exp | 12000             | 0.83      | 0.81   | 0.82    |
