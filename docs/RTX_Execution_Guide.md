# 🚀 RTX GPU Server Execution Guide for GateR-M

이 문서는 학교의 RTX 3090 GPU 머신에서 본 개량 프로젝트(GateR-M)를 이어받아 대규모 HPO(하이퍼파라미터 최적화) 및 전체 벤치마크 훈련을 실행하기 위한 단계별 가이드라인입니다.

---

## 📌 1단계: 소스코드 복제 및 브랜치 전환

GPU 서버에 접속한 후, 아래 명령어를 실행하여 작업 중인 개량 브랜치를 가져옵니다.

```bash
# 1. 저장소 클론 (이미 클론되어 있다면 생략 가능)
git clone https://github.com/chjnett/tabular-dl-tabr.git
cd tabular-dl-tabr

# 2. GateR-M 아키텍처 및 유닛 테스트가 포함된 브랜치 체크아웃
git checkout feature/gate-rm
```

---

## 📌 2단계: 가상환경 격리 구성 및 GPU 가속 라이브러리 설치

서버의 CUDA 드라이버 버전에 알맞은 PyTorch 및 `faiss-gpu`를 설치합니다.

```bash
# 1. 파이썬 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 2. pip 업그레이드 및 기본 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt

# 3. RTX 3090 가속용 PyTorch 및 faiss-gpu 설치 (CUDA 12.1+ 기준)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install faiss-gpu
```

---

## 📌 3단계: GPU 가속 작동성 확인 (Sanity Check)

로컬에서 정의한 11개의 유닛 테스트가 GPU 환경에서 에러 없이 패스하는지 확인합니다.

```bash
# 환경 변수에 프로젝트 루트 등록 후 pytest 실행
export PROJECT_DIR=$(pwd)
pytest tests/
```

> [!NOTE]
> GPU 기기(device.type == 'cuda')로 구동되면, macOS CPU 환경에서 동작하던 cdist L2 fallback 대신 **faiss-gpu 모듈이 내부에서 자동 트리거**되어 극도로 정교하고 빠른 근접 이웃 검색 연산이 실행됩니다.

---

## 📌 4단계: HPO 및 대규모 실험 훈련 가동

### ① 단일 1-Epoch 테스트 훈련 (동작 유효성 최종 확인)
```bash
python bin/tabr.py exp/debug/gate_rm_test.toml --force
```

### ② Optuna HPO 튜닝 파이프라인 백그라운드 구동
작업 도중 세션이 끊겨도 백그라운드에서 HPO가 안정적으로 지속되도록 `nohup` 또는 `tmux` 세션을 이용해 실행합니다.

```bash
# california 데이터셋에 대해 0-tuning.toml 설정을 바탕으로 Optuna 튜닝 시작
nohup python bin/tune.py exp/tabr/california/0-tuning.toml --force > tuning.log 2>&1 &
```

> [!TIP]
> `exp/tabr/california/0-tuning.toml` 파일의 `[space.model]` 섹션 아래에 다음과 같이 신규 튜닝 범위를 개설하여 돌릴 수 있습니다:
> ```toml
> type = "gate_rm"
> d_embedding = ["_tune_", "int", 8, 32]
> n_layers = ["_tune_", "int", 1, 4]
> n_ensembles = ["_tune_", "int", 2, 8]
> ```

---

## 📌 5단계: 최종 SOTA 비교 표 (LaTeX) 파싱

훈련이 완주되면 최적 결과 파일(`report.json`)이 출력 경로에 생성됩니다. LaTeX 포맷터 스크립트를 통해 논문 게재용 테이블 코드를 자동으로 파싱하여 출력합니다.

```bash
python tests/test_phase10_final_report.py
```
