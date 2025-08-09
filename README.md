# 📂 Project 972 - Data Science & AI Portfolio Hub

이 저장소는 제가 수행한 **데이터 분석, 머신러닝, 딥러닝, AI 응용 프로젝트**를 한눈에 볼 수 있는 허브(Hub)입니다.  
각 프로젝트 제목을 클릭하면 GitHub 저장소로 이동하여 상세 코드, 데이터, 보고서를 확인할 수 있습니다.

---

## 📊 [project-statistics](https://github.com/wootae1020/project-statistics)
- **주제:** 통계학 기반 금융 데이터 분석
- **개요:** IT 아웃소싱 플랫폼 L사의 실거래 데이터를 분석하여, 거래 성사율 제고·수수료 구조 개선·고객 유지 전략을 도출
- **주요 활동:**
  - 고객·전문가·서비스·거래 로그 4개 원천 데이터 통합 및 전처리
  - 가설 기반 통계 분석(다중회귀, 로지스틱 회귀, ANOVA, 카이제곱 검정)
  - 가격·전문가 특성·수수료와 거래 성사율의 상관관계 분석
- **성과:**
  - 전문가 평점·응답속도가 만족도 및 거래 성사율에 유의미한 영향
  - 고가 서비스일수록 거래 실패 가능성 증가
  - 수수료율은 매출에는 긍정적이나 성사율과 무관
- **기술 스택:** Python, Pandas, Statsmodels, Matplotlib, Seaborn
- **산출물:**  
  - [기획서 PDF](https://github.com/wootae1020/project-statistics/blob/main/results/statistics_project_proposal.pdf)  
  - [프레젠테이션 PDF](https://github.com/wootae1020/project-statistics/blob/main/results/statistics_project_presentation.pdf)

---

## 🧠 [project-ml](https://github.com/wootae1020/project-ml)
- **주제:** 머신러닝 기반 신용카드 고객 세그먼트 분류
- **개요:** 신용카드 사용 데이터를 기반으로 고객을 A~E 세그먼트로 분류하는 다중 분류 모델을 개발하여, 맞춤형 마케팅 전략 수립 및 비용 효율 개선
- **주요 활동:**
  - Dacon(2018) 및 iM뱅크(2021~2023) 카드 데이터 통합 분석
  - 결측치 처리, 인코딩, 상관관계 기반 피처 선택 및 중복 제거
  - XGBoost·LightGBM·CatBoost·Logistic·Decision Tree 비교
- **성과:**
  - Gradient Boosting 기반 Soft Voting Ensemble로 F1-score 0.649 달성
  - 고차원·불균형 데이터에 대한 안정적 성능 확보
  - 세그먼트 기준 재정의 및 피처 정제 필요성 확인
- **기술 스택:** Python, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna
- **산출물:**  
  - [기획서 PDF](https://github.com/wootae1020/project-ml/blob/main/results/ml_project_proposal.pdf)  
  - [프레젠테이션 PDF](https://github.com/wootae1020/project-ml/blob/main/results/ml_project_presentation.pdf)

---

## 🚇 [project-dl](https://github.com/wootae1020/project-dl)
- **주제:** 지하철 혼잡도 예측 기반 광고 전략 최적화
- **개요:** 서울 지하철 1~8호선의 1시간 단위 혼잡도를 LSTM 모델로 예측하여, 광고 노출 전략 및 요금 차등화에 활용
- **주요 활동:**
  - 2021~2023년 서울교통공사 혼잡도 데이터 + 기상청 기상 데이터 통합
  - 결측치 보간, 이상치 제거, 범주형 인코딩, 시간 주기성 변환 등 전처리
  - LSTM(2층, 64→32유닛) 기반 호선별 모델 학습 및 성능 최적화
- **성과:**
  - 호선별 모델 R² 0.9646~0.978, RMSE 3.14~4.93
  - 황사 발생 시 혼잡도 평균 9.29% 증가 분석
  - 혼잡도 기반 광고 전략 제안 (시간대·노선별 맞춤형 광고)
- **기술 스택:** Python, TensorFlow/Keras(LSTM), Pandas, Scikit-learn, Matplotlib
- **산출물:**  
  - [기획서 PDF](https://github.com/wootae1020/project-dl/blob/main/results/dl_project_proposal.pdf)  
  - [프레젠테이션 PDF](https://github.com/wootae1020/project-dl/blob/main/results/dl_project_presentation.pdf)

---

## 🛒 [project-mldl](https://github.com/wootae1020/project-mldl)
- **주제:** 정형 데이터 기반 AutoML 분석 및 고객 인사이트 도출
- **개요:** 유아용품 전문 쇼핑몰 '맘큐'의 내부 데이터와 외부 데이터를 결합하여 고객 이탈 예측, 구매 패턴 분석, 배송 소요 기간 예측, 브랜드 캠페인 효과 분석 수행
- **주요 활동:**
  - **구매 패턴 분석:** RFM 지표 기반 고객 세그먼트 분류 및 세그먼트별 맞춤 마케팅 전략 제안
  - **고객 이탈 예측:** H2O AutoML로 이탈 확률 예측 모델 개발 및 리텐션 전략 도출
  - **배송일 예측:** 외부 물류 데이터 결합 → LightGBM, AutoML 성능 비교
  - **브랜드 캠페인 분석:** 허그박스 리뷰 데이터 NLP 분석(Okt, Kiwi, LDA)으로 고객 반응 및 만족도 파악
- **성과:**
  - 세그먼트별 차별화된 마케팅 인사이트 도출
  - 이탈 위험 고객 조기 식별 및 대응 전략 마련
  - 배송 지연 사전 감지 가능 모델 구현
- **기술 스택:** Python, pandas, scikit-learn, LightGBM, H2O AutoML, NLP(Okt, Kiwi, LDA), matplotlib, seaborn, plotly
- **산출물:**  
  - [기획서 PDF](https://github.com/wootae1020/project-mldl/blob/main/results/mldl_project_proposal.pdf)  
  - [프레젠테이션 PDF](https://github.com/wootae1020/project-mldl/blob/main/results/mldl_project_presentation.pdf)

---

## 💱 [project-fxrisk](https://github.com/wootae1020/project-fxrisk)
- **주제:** AI 기반 환율 예측 및 환리스크 대응 플랫폼
- **개요:** LSTM 기반 환율 예측과 거시경제·시장 데이터를 결합하여 수출입 기업에 맞춤형 환헤지 전략을 자동 제안하는 통합 웹 서비스 개발
- **주요 활동:**
  - 2022~2025년 USD, JPY, EUR, CNY 환율 데이터 + 기술적 지표 + 거시경제지표 + EPU 지수 + 뉴스 데이터 통합
  - LSTM·XGBoost·Random Forest 모델 비교 및 최적화
  - BERT 기반 뉴스 감성 분석, LDA 토픽 모델링, COSTAR 프롬프트 기반 통화정책 브리핑 요약
- **성과:**
  - USD/KRW 1일 후 예측 R² ≈ 0.9555 (LSTM)
  - 날짜별 주요 이슈·EPU 추이·예측 환율 기반 손익 시뮬레이션 및 전략 추천 기능 구현
  - 기업·은행·투자자가 환율 변동에 선제 대응할 수 있는 실용적 플랫폼 완성
- **기술 스택:** Python, LSTM, XGBoost, Random Forest, BERT, Streamlit, Pandas, Matplotlib, Seaborn
- **산출물:**  
  - [기획서 PDF](https://github.com/wootae1020/project-fxrisk/blob/main/results/fxrisk_project_proposal.pdf)  
  - [프레젠테이션 PDF](https://github.com/wootae1020/project-fxrisk/blob/main/results/fxrisk_project_presentation.pdf)  
  - [보고서 PDF](https://github.com/wootae1020/project-fxrisk/blob/main/results/fxrisk_project_report.pdf)
