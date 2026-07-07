# 🚀 RTX GPU Server Execution Guide for GateR-M

이 문서는 RTX 3090 GPU 머신(Windows + CUDA 12.1+)에서 본 개량 프로젝트(GateR-M)를 이어받아
대규모 HPO(하이퍼파라미터 최적화) 및 전체 벤치마크 훈련을 실행하기 위한 단계별 가이드라인입니다.

> [!NOTE]
> **실행 환경 (검증 완료):** Windows 11 · RTX 3090 24GB · CUDA Driver 13.1 · Anaconda Python 3.10

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

**✅ 검증 결과 (2026-07-07):** `feature/gate-rm` 브랜치 체크아웃 성공

---

## 📌 2단계: Conda 가상환경 격리 구성 및 GPU 가속 라이브러리 설치

> [!IMPORTANT]
> **Windows 환경** 기준입니다. `faiss-gpu`는 Windows pip에서 제공되지 않으므로
> `faiss-cpu`로 대체합니다. RTX 3090의 CUDA를 통해 PyTorch 연산은 GPU에서 실행됩니다.

```powershell
# 1. Python 3.10 conda 환경 생성
conda create -n tabr python=3.10 -y
conda activate tabr

# 2. pip 업그레이드
pip install --upgrade pip

# 3. RTX 3090 가속용 PyTorch 설치 (CUDA 12.1 빌드)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 4. 프로젝트 의존성 설치
pip install numpy optuna pandas scikit-learn scipy loguru tqdm tomli tomli-w delu==0.0.15 einops==0.6.1 rtdl==0.0.13 tensorboard

# 5. faiss-cpu 설치 (Windows에서 faiss-gpu pip 배포 없음)
pip install faiss-cpu pytest

# 6. PyTorch 버전 복구 (delu 설치 후 다운그레이드될 경우 실행)
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

> [!WARNING]
> `delu==0.0.15`는 `torch<2` 의존성을 선언하지만, **실제 코드는 PyTorch 2.x에서 정상 동작**합니다.
> pip 경고는 무시해도 됩니다. 단, 설치 순서상 delu 먼저 설치 시 PyTorch가 다운그레이드될 수 있으니
> **반드시 PyTorch를 마지막에 `--force-reinstall`로 설치**하세요.

**✅ 검증 결과:** `torch 2.5.1+cu121 | CUDA available: True | GPU: NVIDIA GeForce RTX 3090`

---

## 📌 3단계: 데이터셋 준비 및 GPU Sanity Check

### ① California Housing 데이터셋 준비

```powershell
# sklearn으로 데이터셋 즉시 준비 (HuggingFace 전체 아카이브 다운로드 불필요)
$env:PYTHONUTF8="1"
python prepare_data.py
```

`data/california/` 하위에 다음 파일이 생성됩니다:
- `X_num_{train,val,test}.npy` — 수치형 피처 (12384 / 4128 / 4128 샘플)
- `Y_{train,val,test}.npy` — 타겟 (regression)
- `info.json`, `READY`

### ② GPU 가속 환경 확인

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 기대 출력: 2.5.1+cu121 True NVIDIA GeForce RTX 3090
```

### ③ 전체 유닛 테스트 실행 (28개)

```powershell
$env:PYTHONUTF8="1"
$env:PROJECT_DIR="C:\your\path\tabular-dl-tabr"
python -m pytest tests/ -v
```

**✅ 검증 결과 (2026-07-07):**
```
============================= 28 passed in 2.55s ==============================
```

> [!NOTE]
> GPU 기기(`device.type == 'cuda'`)로 구동 시, macOS CPU 환경에서 동작하던
> `cdist L2 fallback` 대신 **faiss 모듈이 내부에서 자동 트리거**되어
> 극도로 정교하고 빠른 근접 이웃 검색 연산이 실행됩니다.

---

## 📌 4단계: HPO 및 대규모 실험 훈련 가동

### ① 단일 1-Epoch 테스트 훈련 (동작 유효성 최종 확인)

```powershell
$env:PYTHONUTF8="1"
$env:PROJECT_DIR="C:\your\path\tabular-dl-tabr"
$env:CUDA_VISIBLE_DEVICES="0"
python bin/tabr.py exp/debug/gate_rm_test.toml --force
```

**✅ 검증 결과 (2026-07-07):**
```
n_parameters = 13221
Epoch 0: 100%|██████████| 49/49 [00:01<00:00, 49.21it/s]
(val) -0.714 (test) -0.715 (loss) 0.68630
🌸 New best epoch! 🌸
'gpus': ['NVIDIA GeForce RTX 3090']
'time': '0:00:01'
```

### ② Optuna HPO 튜닝 파이프라인 가동

`exp/tabr/california/0-tuning.toml`의 `[space.model]` 섹션에 아래 내용을 추가하세요:

```toml
type = "gate_rm"
d_embedding = ["_tune_", "int", 8, 32]
n_layers = ["_tune_", "int", 1, 4]
n_ensembles = ["_tune_", "int", 2, 8]
```

```powershell
# 백그라운드 HPO 실행
$env:PYTHONUTF8="1"; $env:CUDA_VISIBLE_DEVICES="0"
python bin/tune.py exp/tabr/california/0-tuning.toml --force > tuning.log 2>&1
```

---

## 📌 5단계: 최종 SOTA 비교 표 (LaTeX) 파싱

훈련이 완주되면 `exp/debug/gate_rm_test/report.json`이 생성됩니다.

```powershell
$env:PYTHONUTF8="1"; $env:PROJECT_DIR="C:\your\path\tabular-dl-tabr"
python tests/test_phase10_final_report.py
```

**✅ 검증 결과 (2026-07-07):**
```latex
\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\hline
Model & Train Score & Val Score & Test Score & Parameters \\
\hline
GateR-M (gate_rm) & -0.7084 & -0.7145 & -0.7153 & 13221 \\
\hline
\end{tabular}
\caption{Performance comparison of GateR-M.}
\end{table}
```

---

## 📋 전체 실행 결과 요약

| Phase | 내용 | 상태 | 비고 |
|-------|------|------|------|
| 1 | `feature/gate-rm` 브랜치 체크아웃 | ✅ | |
| 2 | `tabr` conda 환경 + PyTorch 2.5.1+cu121 | ✅ | faiss-cpu 대체 |
| 3 | pytest 28개 유닛 테스트 | ✅ | **28/28 PASSED** |
| 4-① | GateR-M 1-Epoch 훈련 | ✅ | val=-0.7145, ~1초 |
| 5 | LaTeX 결과 파싱 | ✅ | report.json 자동 생성 |

### 신규 추가 파일

| 파일 | 설명 |
|------|------|
| `exp/debug/gate_rm_test.toml` | GateR-M 1-epoch 디버그 config |
| `exp/debug/tabr_test.toml` | test_model_step용 config |
| `prepare_data.py` | sklearn으로 California 데이터셋 준비 |
| `tests/test_phase2_modularization.py` | Phase 2 모듈 의존성 분리 테스트 (8개) |
| `tests/test_phase9_hpo_integration.py` | Phase 9 Optuna HPO 통합 테스트 (7개) |
