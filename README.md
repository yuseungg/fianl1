# MRI 뇌 종양 분류 프로젝트

## 프로젝트 설명

이 프로젝트는 MRI 뇌 종양 이미지를 분류하기 위한 머신러닝 모델을 개발한 것입니다. 사용된 알고리즘은 Support Vector Machine (SVM)으로, 그리드 서치를 통해 최적의 하이퍼파라미터를 찾아 성능을 향상시켰습니다. 최적의 모델을 훈련한 후 테스트 데이터에 대한 정확도를 평가하였습니다.

## 데이터셋

훈련 데이터셋은 'tumor_dataset/Training' 폴더에서 각 레이블별로 제공되며, 'glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor' 네 가지 클래스로 구성되어 있습니다.

## 모델 튜닝

SVM 모델의 하이퍼파라미터 튜닝을 위해 그리드 서치를 사용하였습니다. 'C'와 'gamma' 파라미터에 대해 다양한 값들을 실험하여 최적의 조합을 찾았습니다.

```python
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
svm_model = SVC()
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_C = grid_search.best_params_['C']
best_gamma = grid_search.best_params_['gamma']
svm_model_tuned = SVC(C=best_C, gamma=best_gamma)
svm_model_tuned.fit(X_train, y_train)
