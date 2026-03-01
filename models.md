# FireDucks Models CLI 사용 가이드

이 문서는 `models` CLI의 사용 방법을 설명합니다.

## 컬럼 이름 (Column Names)

### 기본 컬럼
- `model_id`: 모델의 고유 식별자
- `model_name`: 모델의 표시 이름
- `provider`: 모델 제공업체 (예: openai, google, anthropic)
- `context_window`: 모델의 컨텍스트 윈도우 크기 (토큰 수)
- `cost_input_per_million` (또는 `input_cost`): 입력 100만 토큰당 비용 (USD)
- `cost_output_per_million` (또는 `output_cost`): 출력 100만 토큰당 비용 (USD)
- `max_tokens`: 모델이 생성할 수 있는 최대 토큰 수

> **참고**: `input_cost`와 `output_cost`는 각각 `cost_input_per_million`와 `cost_output_per_million`의 별칭(alias)으로 사용할 수 있습니다. 
> 예: `-c input_cost`는 `-c cost_input_per_million`과 동일하게 동작합니다.

## 기본 사용법

### 도움말 보기
```bash
uv tool run models --help
```

### 기본 출력 (상위 10개 모델)
```bash
uv tool run models
```

### 출력 행 수 제한하기
```bash
uv tool run models --limit 5
```

## 컬럼 선택

### 특정 컬럼만 보기
```bash
uv tool run models -c model_name -c provider -c context_window
```

### 모든 컬럼 보기
```bash
uv tool run models --all-columns
```

## Providers

### 모든 provider 목록 출력 (comma-separated)

```bash
uv tool run models providers
```

### provider별 모델 개수 포함

```bash
uv tool run models providers --count
```

### provider 이름 검색

```bash
uv tool run models providers --search open
```

## 검색 (Fuzzy Search)

`model_id`/`model_name` 기준으로 퍼지 검색을 수행합니다. 기본 정렬은 퍼지 점수 내림차순입니다.

```bash
uv tool run models search gpt
uv tool run models search gpt --in name
uv tool run models search gpt --provider openai
```

## 필터링

### 1. 정확한 값으로 필터링 (Exact Match)
```bash
# 정확히 'openai'인 제공업체의 모델
uv tool run models -f "provider=openai"

# 특정 모델 ID로 검색
uv tool run models -f "model_id=gpt-4"

# 입력 비용이 정확히 5.0인 모델 (두 가지 방법 모두 가능)
uv tool run models -f "cost_input_per_million=5.0"
uv tool run models -f "input_cost=5.0"  # 별칭 사용
```

### 2. 부분 일치 (Partial Match, 대소문자 구분 없음)
```bash
# 모델 이름에 'gpt'가 포함된 모든 모델
uv tool run models -f "model_name~=gpt"

# 제공업체 이름에 'goo'가 포함된 모든 모델 (google, google-ai 등)
uv tool run models -f "provider~=goo"
```

### 3. 정규식 패턴으로 필터링 (Regex)
```bash
# 'gpt-4'로 시작하는 모든 모델
uv tool run models -f "model_name~^gpt-4"

# 'mini' 또는 'small'이 포함된 모델
uv tool run models -f "model_name~mini|small"
```

### 4. 비교 연산자
```bash
# 컨텍스트 윈도우가 100,000보다 큰 모델
uv tool run models -f "context_window>100000"

# 입력 비용이 10 미만인 모델 (두 가지 방법 모두 가능)
uv tool run models -f "cost_input_per_million<10"
uv tool run models -f "input_cost<10"  # 별칭 사용

# 출력 비용이 20 미만인 모델 (두 가지 방법 모두 가능)
uv tool run models -f "cost_output_per_million<20"
uv tool run models -f "output_cost<20"  # 별칭 사용
```

### 5. 복합 필터링
```bash
# OpenAI의 모델 중 컨텍스트가 100K 이상이고 이름에 'gpt'가 포함된 모델
uv tool run models -f "provider=openai" -f "context_window>100000" -f "model_name~=gpt"

# 입력 비용이 10 미만이고 출력 비용이 20 미만인 모델 (별칭 사용)
uv tool run models -f "input_cost<10" -f "output_cost<20"

# 특정 제공업체의 모델 중 입력 비용이 5 이하이거나 컨텍스트가 200K 이상인 모델
uv tool run models -f "provider=anthropic" -f "input_cost<=5" -f "context_window>=200000"
```

## 정렬

### 특정 컬럼으로 정렬 (오름차순/내림차순)
```bash
# 컨텍스트 윈도우가 큰 순서대로 정렬
uv tool run models --sort "context_window:desc"

# 입력 비용이 저렴한 순서대로 정렬
uv tool run models --sort "input_cost:asc"
```

## 실용적인 예제

### 1. OpenAI 모델 중 컨텍스트 윈도우가 큰 순서대로 보기
```bash
uv tool run models \
  -f "provider=openai" \
  -c model_name -c context_window -c cost_input_per_million \
  --sort "context_window:desc"
```

### 2. 가장 저렴한 모델 5개 보기
```bash
# 정식 컬럼명 사용
uv tool run models \
  -c model_name -c provider -c cost_input_per_million \
  --sort "cost_input_per_million:asc" \
  --limit 5

# 별칭 사용 (동일한 결과)
uv tool run models \
  -c model_name -c provider -c input_cost \
  --sort "input_cost:asc" \
  --limit 5
```

### 3. 특정 기능을 지원하는 모델 찾기
```bash
# 파일 첨부를 지원하는 모델
uv tool run models \
  -f "supports_attachments=True" \
  -c model_name -c provider -c context_window

# 추론(reasoning)을 지원하는 모델
uv tool run models \
  -f "supports_reasoning=True" \
  -c model_name -c provider
```

### 4. 대용량 컨텍스트를 지원하는 모델 찾기
```bash
# 컨텍스트 윈도우가 100K 이상이면서 입력 비용이 10 미만인 모델 (정식 컬럼명 사용)
uv tool run models \
  -f "context_window>=100000" \
  -f "cost_input_per_million<10" \
  -c model_name -c provider -c context_window -c cost_input_per_million \
  --sort "cost_input_per_million:asc"

# 위와 동일하지만 별칭 사용
uv tool run models \
  -f "context_window>=100000" \
  -f "input_cost<10" \
  -c model_name -c provider -c context_window -c input_cost \
  --sort "input_cost:asc"
```

## 출력 예시

```
Model Data (showing 5 of 151 models)
┏━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
│ model_name   │ provider │ context_window │
┡━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ GPT-4.1 Mini │ openai   │ 1047576        │
│ o1-preview   │ openai   │ 128000         │
│ o1-mini      │ openai   │ 128000         │
└──────────────┴──────────┴────────────────┘
```

## 참고사항

- 모든 가격은 1백만 토큰당 비용(USD)입니다.
- `context_window`는 모델이 한 번에 처리할 수 있는 최대 토큰 수를 나타냅니다.
- `--limit` 파라미터를 사용하면 결과 수를 제한할 수 있습니다.
- `--all-columns`를 사용하면 사용 가능한 모든 컬럼을 볼 수 있습니다.
- `input_cost`/`output_cost`는 각각 `cost_input_per_million`/`cost_output_per_million`의 별칭으로, 필터링이나 컬럼 지정 시에 동일하게 사용할 수 있습니다.
- 필터링 시 대소문자를 구분하지 않습니다 (예: `provider=OpenAI`와 `provider=openai`는 동일하게 처리됨).
