# 최적화 학습 프로젝트

최적화 알고리즘을 학습하고 비교하기 위한 Python 프로젝트입니다.

## 프로젝트 구조

```
optimization_study/
├── deterministic/          # 결정론적 최적화 방법
│   ├── gradient_descent.py     # 경사 하강법
│   ├── newton_method.py        # 뉴턴 방법
│   └── bfgs.py                 # BFGS 준뉴턴 방법
│
├── stochastic/             # 확률론적 최적화 방법
│   ├── genetic_algorithm.py    # 유전 알고리즘
│   ├── particle_swarm.py       # 입자 군집 최적화
│   ├── simulated_annealing.py  # 담금질 기법
│   └── differential_evolution.py # 차분 진화
│
├── functions/              # 벤치마크 함수
│   ├── convex.py               # 볼록 함수 (Sphere, Rosenbrock)
│   ├── multimodal.py           # 다봉 함수 (Rastrigin, Ackley, Griewank)
│   └── valley.py               # 골짜기 함수 (Beale, Booth, Himmelblau)
│
├── utils/                  # 유틸리티
│   ├── visualization.py        # 시각화 도구
│   └── metrics.py              # 성능 평가 지표
│
├── outputs/                # 결과 저장
│   ├── figures/                # 그래프 이미지
│   └── results/                # 성능 결과 CSV
│
├── main.py                 # 메인 실행 파일
├── requirements.txt        # 의존성 패키지
├── implementation.md       # 구현 계획서
└── README.md               # 프로젝트 설명 (이 파일)
```

## 설치 방법

```bash
pip install -r requirements.txt
```

## 사용 방법

```bash
python main.py
```

메뉴에서 원하는 작업을 선택하세요:
1. 결정론적 방법 테스트
2. 확률론적 방법 테스트
3. 함수 시각화
4. 전체 비교 실험
5. 모두 실행

## 구현된 최적화 방법

### 결정론적 방법
- **경사 하강법**: 모멘텀, 선탐색 지원
- **뉴턴 방법**: 유한 차분 헤시안 지원
- **BFGS**: Wolfe 조건 선탐색

### 확률론적 방법
- **유전 알고리즘**: 토너먼트 선택, 블렌드 교차
- **입자 군집 최적화**: 관성 감쇠
- **담금질 기법**: 적응형 스텝 크기
- **차분 진화**: rand/1/bin, best/1/bin 전략

## 라이선스

MIT License
