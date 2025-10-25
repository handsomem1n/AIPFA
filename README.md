# 통합 운동 피드백 시스템

BiLSTM 기반 운동 분류 + Mediapipe 자세 분석 + LLM 피드백 통합 시스템

## 📁 프로젝트 구조

```
exercise_feedback_system/
├── exercise_feedback_system.py   # 메인 통합 시스템
├── models/
│   ├── __init__.py
│   ├── bilstm_classifier.py      # BiLSTM 운동 분류 모듈
│   ├── model.pt                  # 학습된 모델 (추가 필요)
│   └── label_encoder.pkl         # Label encoder (추가 필요)
├── feedback/
│   ├── __init__.py
│   ├── pose_analyzer.py          # Mediapipe 자세 분석
│   └── llm_feedback.py           # LLM 피드백 생성
├── data/
│   ├── gt_videos/                # Ground truth 영상
│   │   ├── lunge_gt.mp4
│   │   └── highknees_gt.mp4
│   └── user_videos/              # 사용자 영상
├── output/                       # 결과 출력 디렉토리
├── requirements.txt
└── README.md
```

## 🚀 설치

```bash
pip install -r requirements.txt
```

## 📝 사용법

### 1. 기본 사용

```python
from exercise_feedback_system import ExerciseFeedbackSystem
import os

# 시스템 초기화
system = ExerciseFeedbackSystem(
    model_path="./models/model.pt",
    label_encoder_path="./models/label_encoder.pkl",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# 비디오 처리
result = system.process_video(
    user_video_path="./data/user_videos/user_lunge.mp4",
    output_dir="./output"
)

# 결과 확인
print(f"운동 종류: {result['exercise_type']}")
print(f"신뢰도: {result['confidence']:.2%}")
print(f"\n{result['feedback_text']}")
```

### 2. GT 영상 지정

```python
result = system.process_video(
    user_video_path="./data/user_videos/user_lunge.mp4",
    gt_video_path="./data/gt_videos/lunge_gt.mp4",  # 직접 지정
    output_dir="./output"
)
```

## 🔧 주요 기능

### 1. BiLSTM 운동 분류
- 관절 시퀀스 데이터로 운동 종류 자동 분류
- 런지, 하이니즈 등 다양한 운동 지원

### 2. 자세 분석
- Mediapipe 기반 실시간 관절 추출
- GT vs 사용자 영상 비교
- OKS 점수, 코사인 유사도 계산
- 프레임별 Good/Fast/Slow/NG 분류

### 3. 시각화
- 부위별 색상으로 자세 정확도 표시
- GT와 사용자 영상 좌우 비교
- 피드백 영상 자동 생성

### 4. LLM 피드백
- 운동별 맞춤 프롬프트
- 자연스러운 코치 스타일 피드백
- 개선 방향 구체적 제시

## 📊 출력 결과

- **feedback_video.mp4**: 시각화된 피드백 영상
- **feedback_full.json**: 프레임별 상세 분석 데이터
- **feedback_summary.json**: 전체 요약 통계
- **feedback_text**: LLM 생성 텍스트 피드백

## 🧪 모델 학습 (선택)

bilstm2.py를 참고하여 새로운 데이터로 모델 재학습 가능:

```python
# Train/Test 분리
train_df, test_df = extract_test_train(df)

# 모델 학습
model = PoseBiLSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
# ... 학습 루프

# 저장
torch.save(model.state_dict(), "./models/model.pt")
joblib.dump(le, "./models/label_encoder.pkl")
```

## 🎯 커스터마이징

### 새로운 운동 추가

1. **GT 영상 추가**: `data/gt_videos/`에 영상 추가
2. **프롬프트 작성**: `feedback/llm_feedback.py`에 프롬프트 함수 추가
3. **매핑 등록**: `exercise_feedback_system.py`의 `_get_default_gt()` 수정

```python
def _get_default_gt(self, exercise_type: str) -> str:
    gt_mapping = {
        "lunge": "./data/gt_videos/lunge_gt.mp4",
        "highknees": "./data/gt_videos/highknees_gt.mp4",
        "new_exercise": "./data/gt_videos/new_exercise_gt.mp4",  # 추가
    }
    return gt_mapping.get(exercise_type, "./data/gt_videos/default_gt.mp4")
```

## 🔍 트러블슈팅

### CUDA 메모리 부족
```python
# CPU 모드로 실행
torch.device('cpu')
```

### OpenCV 인코더 에러
```python
# 코덱 변경
cv2.VideoWriter_fourcc(*"avc1")  # 또는 "XVID"
```

## 📄 라이센스

MIT License

## 👥 기여자

- BiLSTM 모델: bilstm2.py
- 자세 분석: pose_feedback.py
- 통합: 팀 공동 작업
