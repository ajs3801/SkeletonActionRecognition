# Skeleton Action Recognition

2022 LG U+ 성균관대학교 소프트웨어융합대학 산학협력프로젝트
The variety methods for recognizing the action based on the feature of human skeleton.

## TREE

> `GRU` : (간단한 설명)

> `LSTM` : Mediapipe로 추출한 스켈레톤 정보를 LSTM의 입력으로 사용하여 운동 동작을 분류합니다.

> `Posture` : It only look one skeleton's coordinate and inference what is current action(posture).

> `SkeketonMHI_ObjectDetection` : detect the Skeleton MHI and recognize the action

> `Transformer` : (간단한 설명)

> `WEB`: Web UI of main model, LSTM

## Common data preprocessing

It extracts the normalized coordinates by using Mediapipe.
