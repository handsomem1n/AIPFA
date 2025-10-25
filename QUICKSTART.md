# 빠른 시작 가이드

## 1️⃣ 환경 설정

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 디렉토리 구조 생성
```bash
mkdir -p data/gt_videos data/user_videos output models
```

### 3. 필수 파일 준비
- `models/model.pt`: 학습된 BiLSTM 모델
- `models/label_encoder.pkl`: Label encoder
- `data/gt_videos/`: Ground truth 영상들 (8개 운동)
  - `squat_gt.mp4` - 스쿼트
  - `lunge_gt.mp4` - 런지
  - `side_lunge_gt.mp4` - 사이드 런지
  - `situp_gt.mp4` - 윗몸일으키기
  - `highknees_gt.mp4` - 하이니즈
  - `bridge_gt.mp4` - 브릿지
  - `cobra_gt.mp4` - 코브라
  - `jumpingjack_gt.mp4` - 점핑잭
- OpenAI API 키

### 4. API 키 설정
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 2️⃣ 실행 방법

### 방법 1: 데모 스크립트 (간단)
```bash
python demo.py \
    --user-video data/user_videos/my_exercise.mp4 \
    --output output
```

### 방법 2: Python 코드 (커스텀)
```python
from exercise_feedback_system import ExerciseFeedbackSystem
import os

system = ExerciseFeedbackSystem(
    model_path="./models/model.pt",
    label_encoder_path="./models/label_encoder.pkl",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

result = system.process_video(
    user_video_path="./data/user_videos/my_exercise.mp4",
    output_dir="./output"
)

print(result['feedback_text'])
```

## 3️⃣ 결과 확인

실행 후 `output/` 디렉토리에서:
- `feedback_video.mp4`: 시각화 영상
- `feedback_summary.json`: 통계 요약
- `feedback_full.json`: 상세 분석 데이터

## 4️⃣ 옵션 설정

### GT 영상 직접 지정
```bash
python demo.py \
    --user-video data/user_videos/my_lunge.mp4 \
    --gt-video data/gt_videos/lunge_gt.mp4 \
    --output output
```

### 모델 경로 변경
```bash
python demo.py \
    --user-video data/user_videos/my_exercise.mp4 \
    --model path/to/custom_model.pt \
    --encoder path/to/custom_encoder.pkl \
    --output output
```

## 트러블슈팅

### "model.pt not found" 에러
**A:** bilstm2.py로 모델을 먼저 학습하거나, 학습된 모델 파일을 `models/` 디렉토리에 복사하세요.

### "OPENAI_API_KEY not set" 에러
**A:** API 키를 환경변수로 설정하거나 `--api-key` 옵션 사용:
```bash
python demo.py --user-video video.mp4 --api-key sk-...
```

### OpenCV 코덱 에러
**A:** `feedback/pose_analyzer.py`의 코덱을 변경:
```pythona
out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (W*2, H))
```

## 📌 지원하는 운동 종류

시스템은 다음 8가지 운동에 대한 피드백을 제공합니다:

| 운동 종류  | 프롬프트 파일 | GT 비디오 파일 |
|------------|-------------|--------------|
| Squat | `squat_prompts.md` | `squat_gt.mp4` |
| Lunge | `lunge_prompts.md` | `lunge_gt.mp4` |
| Side Lunge | `side_lunge_prompts.md` | `side_lunge_gt.mp4` |
| Situp | `situp_prompts.md` | `situp_gt.mp4` |
| High Knees | `highknees_prompts.md` | `highknees_gt.mp4` |
| Bridge | `bridge_prompts.md` | `bridge_gt.mp4` |
| Cobra | `cobra_prompts.md` | `cobra_gt.mp4` |
| Jumping Jack | `jumpingjack_prompts.md` | `jumpingjack_gt.mp4` |
