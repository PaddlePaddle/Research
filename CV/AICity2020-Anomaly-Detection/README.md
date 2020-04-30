# Multi-Granularity Tracking with Modularlized Components for Unsupervised Vehicles Anomaly Detection (CVPRW 2020)

This repository contains source codes of team113 for NVIDIA AICity Challenge
2020 Track 4, and the technical details please refer to the paper
"Multi-Granularity Tracking with Modularlized Components for Unsupervised Vehicles Anomaly Detection" 



Our method obtain the F1-score metric at 0.9855 and the RMSE metric at 4.8737, which ranked first in the Track4 test set of the NVIDIA AI CITY 2020 CHALLENGE.

## Requirements

- Paddle1.7-gpu-post97

- cuda9

- cudnn7.5

## Annotations
The training annotations for detection models are in [here](https://drive.google.com/file/d/1TaZxro8fzKYCOsph4NdREief6jwXxZfM/view?usp=sharing).
Extract one frame every 2 seconds, then choose half of them to annotate. The
corresponding is (annotation_number - 1)\*2\*30 = frame_number


## Step1: Detection Model

#### Train the detection model

```
cd det_code/PaddleDetection
sh train.sh
```

#### Inference Procedure

```
cd det_code/PaddleDetection
sh infer.sh
```



## Step2: Background Modeling

#### Extract background 

```
python bg_code/ex_bg_mog.py
```



## Step3: Extraction of Hypothetical Abnormal Mask

#### Obtain motion-based mask

```
python mask_code/mask_frame_diff.py
```

#### Obtain trajectory-based mask

```
python mask_code/mask_track.py
```

#### Mask Fusion

```
python mask_code/mask_fuse.py
```



## Step4: Box-level Tracking

#### Tube construction

```
python box_track/tube_construction.py
```

#### Box_level Tracking

```
python box_track/box_level_tracking.py
```



## Step5: Pixel-level Tracking

#### Coarse anomaly  result for Pixel-level Tracking

```
python pixel_track/coarse_ddet/pixel-level_tracking.py
```

#### Similarity filtering for the preliminary abnormal candidate results

```
python pixel_track/post_process/similar.py
```

#### Backtrack the start time

```
python pixel_track/post_process/time_back.py
```

#### Merge in the temporal dimension

```
python pixel_track/post_process/id.py
```



## Step6: Fusion and Backtracking Optimization

#### Fusion of box_level tracking and pixel-level tracking, and backtracking

```
python fusion_code/fusion_backtracking.py
```

#### If you have any questions or issues in using this code, please feel free to
contact us (liyingying05@baidu.com or wujie23@mail2.sysu.edu.cn)
