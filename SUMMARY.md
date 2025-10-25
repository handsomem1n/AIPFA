## 📦 통합된 구조

```
exercise_feedback_system/
│
├── 📄 exercise_feedback_system.py  ← 메인 통합 시스템
├── 📄 demo.py                      ← 간단한 실행 스크립트
├── 📄 test_setup.py                ← 설치 확인 스크립트
│
├── 📂 models/                      ← BiLSTM 분류 모듈
│   ├── bilstm_classifier.py
│   ├── model.pt                    (추가 필요)
│   └── label_encoder.pkl           (추가 필요)
│
├── 📂 feedback/                    ← 자세 분석 & 피드백
│   ├── pose_analyzer.py
│   └── llm_feedback.py
│
├── 📂 data/                        ← 데이터 디렉토리
│   ├── gt_videos/
│   └── user_videos/
│
├── 📂 output/                      ← 결과 출력
│
└── 📚 문서들
    ├── README.md                   ← 전체 개요
    ├── QUICKSTART.md               ← 빠른 시작
    ├── ARCHITECTURE.md             ← 아키텍처
    └── CHECKLIST.md                ← 체크리스트
```

## 동작 흐름

```
사용자 영상 입력
    ↓
관절 추출 (Mediapipe)
    ↓
운동 분류 (BiLSTM) ← 새로운 기능!
    ↓
GT 영상 자동 선택
    ↓
자세 분석 & 시각화
    ↓
LLM 피드백 생성
    ↓
결과 출력 (영상 + JSON + 텍스트)
```

## 🚀 바로 시작하기

### 1. 설치
```bash
cd exercise_feedback_system
pip install -r requirements.txt
```

### 2. 실행
```bash
# 방법 1: 데모 스크립트
python demo.py --user-video path/to/video.mp4

# 방법 2: Python 코드
from exercise_feedback_system import ExerciseFeedbackSystem
system = ExerciseFeedbackSystem(...)
result = system.process_video(...)
```

## 📝 필요한 작업

1. **모델 파일 준비**
   ```bash
   # bilstm2.py로 학습하거나 기존 파일 복사
   cp /path/to/model.pt models/
   cp /path/to/label_encoder.pkl models/
   ```

2. **GT 영상 추가**
   ```bash
   cp /path/to/lunge_gt.mp4 data/gt_videos/
   cp /path/to/highknees_gt.mp4 data/gt_videos/
   ```

3. **API 키 설정**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

