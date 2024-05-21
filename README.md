## Spectrogram segmentation for underwater noise classification 

We use transfer learning on yolov7 to detect bounding boxes around potential dolphin whistles.

### Results on REPMUS datasets

| Data Type        | Training Labels   | Precision | Recall | mAP@0.5 |
|------------------|-------------------|-----------|--------|---------|
| REP23            | 430               | 0.59      | 0.50   | 0.31    |
| REP23+aug        | 1300              | 0.66      | 0.69   | 0.55    |
| REP23+21+aug     | 5000              | 0.79      | 0.69   | 0.77    |
