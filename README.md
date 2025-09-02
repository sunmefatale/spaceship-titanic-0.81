# 🚀 Kaggle - Spaceship Titanic (캐글 우주 타이타닉)

**목표:** 승객이 다른 차원으로 _Transported_ 되었는지 예측하는 이진 분류 문제  
**최종 성능:** **0.81** (LightGBM + XGBoost 앙상블 + Threshold=0.41)

---

## 📂 프로젝트 구조 예시
spaceship-titanic/
├─ data/ # (선택) 원본 csv. Kaggle에서 내려받아 이 폴더에 넣기
├─ images/ # README에 넣을 캡처 이미지 (선택)
├─ notebooks/
│ └─ spaceship_titanic.ipynb
├─ src/
│ ├─ features.py # 피처 엔지니어링/인코딩 파이프라인 (공통)
│ ├─ model_lgbm.py # LightGBM 학습/예측
│ ├─ model_xgb.py # XGBoost 학습/예측
│ └─ ensemble.py # 앙상블 + 임계값 튜닝 + 제출 파일 생성
├─ submission.csv # 최종 제출 파일 (예시)
└─ README.md

> 폴더까지 꼭 맞출 필요는 없고, 핵심은 **train/test에 동일 파이프라인을 적용**하고,  
> **두 모델이 같은 입력 컬럼(X)으로 학습**하도록 관리하는 것입니다.

---

## 📊 데이터
- `train.csv` (8693 x 14, 타깃 `Transported` 포함)  
- `test.csv`  (4277 x 13, 타깃 없음 — 제출용)  
- `sample_submission.csv` (형식 예시)

👉 Kaggle 대회 링크: [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)

---

## ⚙️ 환경 세팅

### (A) 필수 패키지 설치
```bash
# 권장: 새 가상환경에서 진행 (선택)
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# 공통 패키지 설치
pip install -U pandas numpy scikit-learn lightgbm xgboost
pip install jupyter
jupyter notebook notebooks/spaceship_titanic.ipynb

🧹 전처리 & 피처 엔지니어링

결측치 처리

숫자형(Age) → 중앙값

금액(RoomService~VRDeck) → 0

범주형(HomePlanet, Destination, CryoSleep, VIP) → 최빈값

인코딩: 원-핫 인코딩 (훈련/테스트 동일 규칙 적용)

피처 엔지니어링

GroupSize, IsAlone (PassengerId 기반 그룹 크기)

Deck, Side (Cabin 분해)

TotalSpend, LogTotalSpend, NoSpend, CryoSpendMismatch

(선택) AgeBin, SpendCount, Route(HomePlanet×Destination)

import pandas as pd, numpy as np

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Group features
    out['GroupId']   = out['PassengerId'].str.split('_').str[0]
    out['GroupSize'] = out.groupby('GroupId')['PassengerId'].transform('count')
    out['IsAlone']   = (out['GroupSize']==1).astype(int)
    # Cabin split
    deck = out['Cabin'].str.split('/').str[0]
    side = out['Cabin'].str.split('/').str[2]
    out['Deck'] = deck.fillna('Unknown')
    out['Side'] = side.fillna('Unknown')
    # Spend features
    spend = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    out[spend] = out[spend].fillna(0)
    out['TotalSpend']    = out[spend].sum(axis=1)
    out['LogTotalSpend'] = np.log1p(out['TotalSpend'])
    out['NoSpend']       = (out['TotalSpend']==0).astype(int)
    # Basics
    out['Age'] = out['Age'].fillna(out['Age'].median())
    for c in ['HomePlanet','Destination','CryoSleep','VIP']:
        out[c] = out[c].fillna(out[c].mode()[0])
    # Consistency check
    out['CryoSpendMismatch'] = ((out['CryoSleep']==True) & (out['TotalSpend']>0)).astype(int)
    return out

🤖 모델링 결과
모델	특징	정확도(hold-out)
RandomForest	기본	~0.77
XGBoost	튜닝	~0.79
LightGBM	피처 추가 + early stopping	~0.808
Ensemble(LGBM+XGB) + Threshold=0.41	확률 평균	~0.817 (0.81)

🛠️ 앙상블 에러 방지 체크리스트

feature_names mismatch 오류를 피하려면:

1.train/test에 같은 함수로 피처 엔지니어링 적
2.원-핫 인코딩 후 열 맞추기
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)
3.LGBM/XGBM 모두 같은 X 데이터셋으로 학습
4.인코딩 옵션(drop_first=True)도 반드시 동일하게 적용
5.파생 피처는 훈련/테스트/두 모델 모두 동일하게 적용

🎯 최종 제출 코드
# 앙상블 예측 + 임계값 적용
p_test = (lgbm.predict_proba(X_test)[:,1] + xgb.predict_proba(X_test)[:,1]) / 2
pred = (p_test >= 0.41).astype(bool)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": pred
})
submission.to_csv("submission.csv", index=False)
print("✅ Saved submission.csv")

📈 결과 요약

RandomForest → 0.77

XGBoost → 0.79

LightGBM(+피처 엔지니어링) → 0.808

앙상블 + Threshold 튜닝 → 0.817 (~0.81)

📚 배운 점

단일 모델은 한계 → 앙상블로 성능 향상

Threshold 조정만으로도 점수 개선 가능

좋은 피처 엔지니어링이 캐글 성능 개선의 핵심

📜 라이선스

MIT License (자유롭게 수정/재배포 가능)
