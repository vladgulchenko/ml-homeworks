# Classical ML Homeworks

Репозиторий с Jupyter-ноутбуками по классическому машинному обучению. Основной формат работ здесь не “конспект по теме”, а небольшие самостоятельные разборы: данные, очистка, признаки, модель, валидация и сравнение результатов.

Все ноутбуки лежат в исходном порядке домашних работ, но читаются как отдельные кейсы по классическому ML.

## Ключевые ноутбуки

Если открывать репозиторий не подряд, а с самых показательных работ, то лучше начать с этих ноутбуков:

- [`hw04_linear/hw04_linear.ipynb`](hw04_linear/hw04_linear.ipynb)  
  Сквозной регрессионный пайплайн на датасете Toyota Corolla: очистка данных, EDA, кодирование признаков, feature engineering и сравнение `LinearRegression`, `Ridge`, `Lasso`.

- [`hw06_bayes/naive_bayes.ipynb`](hw06_bayes/naive_bayes.ipynb)  
  Гауссовский Наивный Байес на Iris: реализация `MyGaussianNB` с нуля, оценка априорных вероятностей, средних и дисперсий, расчет log-likelihood, confusion matrix, сравнение со `sklearn.GaussianNB` и разбор допущения о независимости признаков.

- [`hw07_decision_tree/decision-tree.ipynb`](hw07_decision_tree/decision-tree.ipynb)  
  Деревья решений для классификации: переобучение, pre-pruning, post-pruning, подбор гиперпараметров и работа с пропусками.

- [`hw08_svm/svm_practice_student.ipynb`](hw08_svm/svm_practice_student.ipynb)  
  Метод опорных векторов на Breast Cancer Wisconsin: сравнение SVM с масштабированием и без, `Pipeline`, `GridSearchCV`, анализ ошибок, confusion matrix и визуализация границы решений.

- [`hw09_forest/hw09_forest.ipynb`](hw09_forest/hw09_forest.ipynb)  
  Продолжение Toyota-кейса с ансамблями: `RandomForestRegressor`, ручной подбор параметров, кросс-валидация и итоговое сравнение с `Ridge`.

- [`hw10_gb/hw10_gb.ipynb`](hw10_gb/hw10_gb.ipynb)  
  Градиентный бустинг для регрессии: сравнение XGBoost, LightGBM и CatBoost, а также доработка собственной реализации бустинга с `subsample`, `colsample_bytree`, `feature_importances_` и обработкой категориальных признаков.

- [`hw11_hp/hp_interp.ipynb`](hw11_hp/hp_interp.ipynb)  
  Подбор гиперпараметров и интерпретируемость на Breast Cancer Wisconsin: baseline через 5-fold CV, `GridSearchCV` для SVM, `RandomizedSearchCV` для RandomForest, честная финальная оценка на test, Permutation Importance, PDP/ICE, SHAP и диагностика утечек через `leaky_feature`.

## Структура репозитория

- [`hw01_setup_tools`](hw01_setup_tools/hw01_setup_tools.ipynb) — базовая настройка окружения, NumPy / Pandas, простые визуализации.
- [`hw02_eda`](hw02_eda/hw02_eda.ipynb) — EDA на Titanic.
- [`hw03_knn`](hw03_knn/hw03_knn.ipynb) — KNN на Iris с разбиением `train / validation / test`.
- [`hw04_linear`](hw04_linear/hw04_linear.ipynb) — линейная регрессия на Toyota Corolla.
- [`hw05_logistic`](hw05_logistic/logreg_practice.ipynb) — логистическая регрессия с нуля и сравнение со `scikit-learn`.
- [`hw06_bayes`](hw06_bayes/naive_bayes.ipynb) — Гауссовский Наивный Байес с нуля на Iris и сравнение со `scikit-learn`.
- [`hw07_decision_tree`](hw07_decision_tree/decision-tree.ipynb) — деревья решений для классификации.
- [`hw08_svm`](hw08_svm/svm_practice_student.ipynb) — SVM для бинарной классификации опухолей: `LinearSVC`, `SVC`, масштабирование признаков, подбор `C`, `kernel`, `gamma`.
- [`hw09_forest`](hw09_forest/hw09_forest.ipynb) — случайный лес для регрессии на Toyota Corolla.
- [`hw10_gb`](hw10_gb/hw10_gb.ipynb) — градиентный бустинг для регрессии на California Housing, собственная реализация `MyBoost` и сравнение с XGBoost / LightGBM / CatBoost.
- [`hw11_hp`](hw11_hp/hp_interp.ipynb) — подбор гиперпараметров, сравнение моделей и интерпретация RandomForest через PI, PDP/ICE и SHAP.

## Что покрывают работы

- разведочный анализ данных и очистку признаков;
- обработку пропусков и mixed-type значений;
- one-hot encoding и feature engineering;
- разбиение выборки, кросс-валидацию и `GridSearchCV`;
- сравнение моделей на одинаковой постановке задачи;
- построение пайплайнов с масштабированием признаков;
- реализацию Наивного Байеса с нуля: priors, Gaussian likelihood, log-likelihood и предсказание по максимуму апостериорного score;
- анализ матрицы ошибок и ошибок классификации;
- проверку допущений модели через корреляции признаков внутри классов;
- визуализацию границ решений для классификаторов;
- ручную реализацию ансамблевых методов и проверку их свойств;
- стохастический градиентный бустинг: подвыборка объектов, подвыборка признаков, важности признаков;
- подбор гиперпараметров через `RandomizedSearchCV`;
- интерпретацию моделей через `feature_importances_`, Permutation Importance, PDP/ICE и SHAP;
- диагностику подозрительных признаков, утечек target и шумовых переменных;
- оценку качества через `RMSE`, `MAE`, `R^2`, `accuracy`, `F1`, `ROC-AUC` и другие стандартные метрики.

По моделям в репозитории есть:
- KNN;
- линейная регрессия, `Ridge`, `Lasso`;
- логистическая регрессия;
- Гауссовский Наивный Байес: собственная реализация `MyGaussianNB` и `sklearn.GaussianNB`;
- SVM: `LinearSVC`, `SVC` с линейным и RBF-ядром;
- деревья решений;
- случайный лес;
- градиентный бустинг, включая собственную реализацию и библиотечные XGBoost, LightGBM, CatBoost.

## Окружение

Основные зависимости перечислены в [`requirements.txt`](requirements.txt):

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

Для ноутбука [`hw10_gb`](hw10_gb/hw10_gb.ipynb) дополнительно используются:

- XGBoost
- LightGBM
- CatBoost

Для ноутбука [`hw11_hp`](hw11_hp/hp_interp.ipynb) дополнительно используются:

- SHAP

Датасеты, используемые в ноутбуках, уже лежат внутри соответствующих папок или загружаются через `sklearn.datasets`, поэтому после установки зависимостей репозиторий можно запускать локально без дополнительной подготовки данных.

## Запуск

Если окружение уже собрано, достаточно открыть нужный ноутбук в Jupyter.

Если нужно поднять окружение с нуля:

```bash
pip install -r requirements.txt
jupyter notebook
```

Для `hw11_hp`, если SHAP или Optuna не установлены:

```bash
pip install shap optuna
```

Если используется Conda, достаточно активировать свое окружение перед установкой зависимостей или запуском Jupyter.
