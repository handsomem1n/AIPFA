# 프롬프트 관리 가이드

이 디렉토리는 운동별 피드백 프롬프트를 관리합니다. MD 파일로 프롬프트를 관리하여 코드 수정 없이 쉽게 내용을 변경할 수 있습니다.

## 파일 명명 규칙

운동 종류에 따라 프롬프트 파일명이 자동으로 매핑됩니다:

- **런지** → `lunge_prompts.md`
- **하이니즈** → `highknees_prompts.md`
- **사이드 런지** → `side_lunge_prompts.md`
- **스쿼트** → `squat_prompts.md`

### 파일명 생성 규칙

1. 운동명을 공백으로 분리
2. 각 단어를 영문으로 변환 (한글-영문 매핑 사용)
3. 단어를 언더스코어(`_`)로 연결
4. `_prompts.md` 접미사 추가

**예시:**
- "사이드 런지" → "side" + "lunge" → `side_lunge_prompts.md`
- "프론트 스쿼트" → "front" + "squat" → `front_squat_prompts.md`

## 프롬프트 파일 구조

각 프롬프트 파일은 다음 섹션들로 구성됩니다:

### 필수 섹션

#### `## exercise_description`
운동에 대한 기본 설명

```markdown
## exercise_description
런지는 시상면에서의 고관절 힌지와 슬관절 굴곡/신전의 협응을 요구하며,
체간-경골 각도의 조화와 전방 무릎 이동, 전발 뒤꿈치 접지 유지가 정렬 안정성의 핵심이다.
```

#### `## front_orientation`
정면에서 촬영했을 때의 자세 기준

```markdown
## front_orientation
** 정면에서 바라봤을 때 런지 자세에 대한 설명
  1) 무릎 정렬: 무릎이 두 번째 발가락 방향과 일직선을 유지
  2) 골반 수평: 양쪽 엉덩이가 수평을 이루어야 함
  ...
```

#### `## side_orientation`
측면에서 촬영했을 때의 자세 기준

```markdown
## side_orientation
** 측면에서 바라봤을 때 런지 자세에 대한 설명
  1) 경골 vs 체간 각도: 하강 구간에서 경골과 체간이 유사한 기울기를 유지
  2) 전방 무릎 이동: 전발 무릎은 발끝을 과도하게 초과하지 않도록 한다.
  ...
```

#### `## joint_descriptions`
주요 관절에 대한 설명 (MediaPipe 랜드마크 기준)

```markdown
## joint_descriptions
** 주요 관절에 대한 설명
	• 11: 왼쪽 어깨
	• 12: 오른쪽 어깨
	• 23: 왼쪽 엉덩이
	• 24: 오른쪽 엉덩이
	• 25: 왼쪽 무릎
	• 26: 오른쪽 무릎
	...
```

#### `## evaluation_criteria`
평가 기준 점수 범위

```markdown
## evaluation_criteria
** 평가 기준 (엄격 모드):
- joint_score >= 0.999 : 매우 잘함
- 0.985 ~ 0.997 : 부족
- 0.975 ~ 0.985 : 매우부족
- < 0.975 : 잘못됨
```

#### `## output_guidelines`
LLM이 피드백을 생성할 때 따라야 할 지침

```markdown
## output_guidelines
# 출력 지침:
1. 런지의 핵심 동작 기준과 비교해서 설명한다.
2. 주요 관절의 결과들을 보고 정확하게 피드백을 해준다.
3. 전체적인 동작 평가 (라벨 비율 기반).
4. 개선할 수 있는 구체적인 조언을 제시.
```

### 선택 섹션

#### `## joint_score_interpretation`
관절 점수 해석 규칙 (필요한 경우)

```markdown
## joint_score_interpretation
** 관절 점수 해석 규칙 **
- 23-25, 24-26: 무릎 높이를 반영
- 23-24: 좌우 중심 불균형
- 25-27, 26-28: 착지나 다리 라인 불안정
```

## 플레이스홀더

출력 지침에서 사용할 수 있는 플레이스홀더:

- `{exercise}`: 운동 종류로 자동 치환 (예: "런지", "사이드 런지")
- `{orientation}`: 촬영 방향으로 자동 치환 (예: "front", "side")

**예시:**
```markdown
## output_guidelines
# 출력 지침:
- {orientation} 기준으로 {exercise}의 핵심 동작과 비교하여 피드백한다.
```

## 새 운동 추가하기

### 1. 프롬프트 파일 생성

`prompts/` 디렉토리에 새 MD 파일을 생성합니다.

**파일명 예시:**
- 스쿼트: `squat_prompts.md`
- 버피: `burpee_prompts.md`
- 플랭크: `plank_prompts.md`

### 2. 섹션 작성

위에 설명된 섹션 구조를 따라 내용을 작성합니다.

### 3. 코드에서 사용

```python
from feedback.llm_feedback import FeedbackGenerator

generator = FeedbackGenerator(api_key="your-api-key")

# 자동으로 squat_prompts.md 파일을 로드
feedback = generator.generate_feedback(
    exercise_type="스쿼트",
    summary_data=analysis_result,
    orientation="front"
)
```

## 기본 프롬프트

운동별 프롬프트 파일이 없는 경우, `default_prompts.md`가 사용됩니다.

## 한글-영문 매핑

현재 지원되는 한글-영문 변환:

```python
{
    '런지': 'lunge',
    '하이니즈': 'highknees',
    '스쿼트': 'squat',
    '사이드': 'side',
    '프론트': 'front',
    '백': 'back',
}
```

새로운 운동을 추가할 때 필요한 경우 [llm_feedback.py](../feedback/llm_feedback.py)의 `_get_prompt_filename()` 메서드에서 매핑을 추가할 수 있습니다.

## 프롬프트 캐싱

프롬프트 파일은 메모리에 캐싱되어 반복 호출 시 성능이 최적화됩니다.

## 예시 파일

- [lunge_prompts.md](lunge_prompts.md) - 런지 프롬프트
- [highknees_prompts.md](highknees_prompts.md) - 하이니즈 프롬프트
- [side_lunge_prompts.md](side_lunge_prompts.md) - 사이드 런지 프롬프트
- [default_prompts.md](default_prompts.md) - 기본 프롬프트
