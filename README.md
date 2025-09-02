# 🚀 Kaggle - Spaceship Titanic (캐글 우주 타이타닉)

**목표:** 승객이 다른 차원으로 _Transported_ 되었는지 예측하는 이진 분류 문제  
**최종 성능:** **0.81** (LightGBM + XGBoost 앙상블 + Threshold=0.41)

---

## 📂 프로젝트 구조

```
spaceship-titanic/
├─ train.csv              # 학습 데이터 (Kaggle 제공)
├─ test.csv               # 테스트 데이터 (Kaggle 제공)
├─ sample_submission.csv  # 제출 형식 예시 (Kaggle 제공)
├─ spaceship_titanic.ipynb  # 실험/분석 노트북
├─ submission.csv         # 최종 제출 파일
└─ README.md              # 프로젝트 설명 문서
```

---

## 📊 데이터
- `train.csv` (8693 x 14, 타깃 `Transported` 포함)  
- `test.csv`  (4277 x 13, 타깃 없음 — 제출용)  
- `sample_submission.csv` (형식 예시)

👉 Kaggle 대회 링크: [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)

---

## ⚙️ 환경 세팅

### 1) 필수 패키지 설치
```bash
pip install -U pandas numpy scikit-learn lightgbm xgboost
```

### 2) 노트북 실행
```bash
pip install jupyter
jupyter notebook spaceship_titanic.ipynb
```

---

## 🧹 전처리 & 피처 엔지니어링

- 결측치 처리  
  - Age → 중앙값  
  - RoomService~VRDeck → 0  
  - 범주형(HomePlanet, Destination, CryoSleep, VIP) → 최빈값  

- 인코딩: 원-핫 인코딩 (훈련/테스트 동일 규칙 적용)  

- 파생 피처 예시  
  - GroupSize, IsAlone (PassengerId 기반 그룹 크기)  
  - Deck, Side (Cabin 분해)  
  - TotalSpend, LogTotalSpend, NoSpend, CryoSpendMismatch  

---

## 🤖 모델링 결과

| 모델 | 특징 | 정확도 (hold-out) |
|------|------|-------------------|
| RandomForest | 기본 | ~0.77 |
| XGBoost | 튜닝 | ~0.79 |
| LightGBM | 피처 추가 + early stopping | ~0.808 |
| **Ensemble (LGBM+XGB) + Threshold=0.41** | 확률 평균 | **~0.817 (~0.81)** |

---

## 🛠️ 앙상블 에러 방지 체크리스트
- train/test에 같은 함수로 피처 엔지니어링 적용  
- 원-핫 인코딩 후 열 맞추기  

```python
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)
```

- 두 모델 모두 같은 X 데이터셋으로 학습  
- 인코딩 옵션(drop_first=True) 동일 적용  
- 파생 피처는 train/test 모두 동일하게 생성  

---

## 🎯 최종 제출 코드
```python
# 앙상블 예측 + 임계값 적용
p_test = (lgbm.predict_proba(X_test)[:,1] + xgb.predict_proba(X_test)[:,1]) / 2
pred = (p_test >= 0.41).astype(bool)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": pred
})
submission.to_csv("submission.csv", index=False)
print("✅ Saved submission.csv")
```

---

## 📈 결과 요약
- RandomForest → 0.77  
- XGBoost → 0.79  
- LightGBM(+피처 엔지니어링) → 0.808  
- **앙상블 + Threshold 튜닝 → 0.817 (~0.81)** ✅  

---

## 📚 배운 점
- 단일 모델은 한계 → 앙상블로 성능 향상  
- Threshold 조정만으로도 점수 개선 가능  
- 좋은 피처 엔지니어링이 캐글 성능 개선의 핵심  

---

## 📜 라이선스
MIT License (자유롭게 수정/재배포 가능)
