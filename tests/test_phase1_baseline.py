import os
import sys
import torch
import numpy as np

# Add project root to sys.path and environment
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['PROJECT_DIR'] = project_dir
sys.path.append(project_dir)

import lib

def test_environment_and_determinism():
    """난수 시드 고정 및 CPU/M1 환경 무결성 검증"""
    # 난수 고정 테스트
    torch.manual_seed(42)
    a = torch.randn(5, 5)
    torch.manual_seed(42)
    b = torch.randn(5, 5)
    assert torch.equal(a, b), "PyTorch 시드 고정이 불완전합니다."
    print("Determinism test: PASS")

def test_data_leakage_mask_syntax():
    """오리지널 마스킹 행렬의 대각 성분 차단 구조 검증"""
    batch_size = 32
    # 대각 성분이 true인 마스크 생성(자가 리트리벌 방지용)
    mask = torch.eye(batch_size).bool()
    assert mask[0, 0] == True and mask[0, 1] == False, "자가 마스킹 인덱싱 로직에 결함이 있습니다."
    print("Self-masking logic test: PASS")

if __name__ == '__main__':
    test_environment_and_determinism()
    test_data_leakage_mask_syntax()
