## LSTM

congfigs.py : 대부분의 코드 파일들에서 공유하는 변수들 정의

extrafeatures.py : 미디어파이프로 추출한 스켈레톤 좌표를 모델의 입력에 맞는 형식의 스켈레톤 좌표 + 각도로 변환
extract_v3.py : 동영상 파일에서 미디어파이프를 사용하여 스켈레톤 좌표를 extrafeatures.py를 이용하여 모델의 입력에 맞는 형식으로 추출하여 저장, 저장시 opencv를 사용하여 8가지 augmentation 적용 후 스켈레톤 추출
makedataset_v3.py
