## LSTM

### congfigs.py : 
- 대부분의 코드 파일들에서 공유하는 변수들 정의

### extrafeatures.py : 
- 미디어파이프로 추출한 스켈레톤 좌표를 모델의 입력에 맞는 형식의 스켈레톤 좌표 + 각도로 변환

### extract_v3.py : 
- 동영상 파일에서 미디어파이프를 사용하여 스켈레톤 좌표를 extrafeatures.py를 이용하여 모델의 입력에 맞는 형식으로 추출하여 저장
- 저장시 opencv를 사용하여 8가지 augmentation 적용 후 스켈레톤 추출하여 npy 파일로 저장
- 비디오 하나당 8개의 npy 파일 생성

### sampling20.py :
- 모든 동영상은 30프레임으로 수집되었음
- 데이터셋을 만들때 시퀀스 길이가 30인 파일을 샘플링하여 시퀀스 길이 20인 파일로 augmentation 하기 위한 함수들 정의

### noise.py :
- 스켈레톤 좌표, 각도에 정규분포에서 추출한 노이즈 값을 더하여 augmentation

### makedataset_v3.py :
- extract_v3.py로 생성한 npy 파일을 train : valid : test = 8 : 1 : 1 비율로 분할
- 분할한 뒤 각각의 파일에 대하여 시퀀스에 대한 augmentation과 노이즈를 더해주는 augmentation을 진행하여 train, valid, test 데이터셋 생성

### model_v3.py :
- 전체 스켈레톤을 왼쪽, 오른쪽 그리고 상체, 하체를 기준으로 4가지 파트로 나누어 계층적으로 처리하는 LSTM 모델

### train_v3.py :
- 모델을 학습시키기 위한 하이퍼 파라미터 정의
- 모델 학습 진행
- 가장 높은 valid acc를 기록한 epoch의 모델이 best_model 폴더에 저장됨

### count.py :
- 운동을 카운트하기 위한 플래그 변수 및 함수들 정의

### engine.py :
- 운동 개수 카운트 엔진
