🧠 Machine Learning Projects by Egor Dudov
  Коллекция учебных и прикладных проектов по машинному обучению, глубинному обучению и анализу данных.
  Каждая папка — отдельная законченная работа с собственным пайплайном, датасетом и результатами.
  🚀 Цель репозитория
  Показать навыки в построении ML-пайплайнов:от предобработки данных до построения и деплоя моделей в продакшен (через FastAPI, PyTorch, TensorFlow).
  📚 Проекты
1. 🧩 BinaryClassificator
  Задача: Бинарная классификация
  Методы	Logistic Regression, Random Forest
  Особенности	Нормализация признаков, подбор гиперпараметров
  Метрики	Accuracy, ROC-AUC, F1-score
  📘 Демонстрирует базовый ML-пайплайн: очистка данных, обучение, валидация, интерпретация.
2. 🤖 Contradictory (NLP)
  Задача: Определение противоречий между текстами (Natural Language Inference)
  Датасет	Kaggle / Contradictory MNLI
  Модель	BERT (HuggingFace Transformers)
  Особенности	Fine-tuning, токенизация, паддинг, постобработка
  Метрики	Accuracy, F1-score
  🧠 Основной упор — работа с текстом и архитектурой Transformer.
3. 💳 CreditScorring
  Задача: Предсказать вероятность дефолта клиента
  Датасет	Финансовые данные клиентов
  Модели	Logistic Regression, XGBoost
  Особенности	One-Hot и Label Encoding, балансировка классов
  Метрики	ROC-AUC, Precision-Recall, Confusion Matrix
  💼 Демонстрация работы с категориальными данными и интерпретации признаков.
4. 🔢 DigitRecognizer
  Задача: Распознавание рукописных цифр (MNIST)
  Датасет	MNIST
  Модель	CNN (Keras / TensorFlow)
  Особенности	Dropout, Batch Normalization
  Метрики	Accuracy ≈ 0.98
  🧩 Классический пример обучения сверточной нейронной сети.
5. 🌸 Flower_class
  Задача: Классификация изображений цветов
  Датасет	Flower Recognition Dataset
  Модель	Transfer Learning (ResNet, EfficientNet)
  Особенности	Аугментация, регуляризация
  Метрики	Accuracy, F1-score
  🎨 Пример применения компьютерного зрения и transfer learning.
6. 🎯 Multi-ClassPrediction
  Задача: Многоклассовая классификация
  Датасет	Синтетический / открытый
  Модели	DecisionTree, RandomForest, LightGBM
  Особенности	Обработка несбалансированных данных
  Метрики	Macro-F1, Accuracy
  📊 Сравнение классических моделей классификации.
7. 💬 ProcessingWithDisasterTweets
  Задача: Определение, относится ли твит к реальной катастрофе
  Датасет	Kaggle “Disaster Tweets”
  Модели	Logistic Regression, LSTM
  Особенности	Токенизация, TF-IDF, Embeddings
  Метрики	F1-score, Accuracy
  🌍 Проект по NLP с акцентом на обработку естественного языка и текстовых признаков.
8. 🏠 SalePrice
  Задача: Прогнозирование цены дома
  Датасет	Kaggle “House Prices”
  Модели	Ridge, Lasso, XGBoost
  Особенности	Feature Engineering, кросс-валидация
  Метрики	RMSE, R²
  📈 Практика регрессии и анализа факторов, влияющих на цену.
9. 🚀 SpaceshipTitanic
  Задача: Классификация пассажиров (спасён / не спасён)
  Датасет	Kaggle “Spaceship Titanic”
  Модели	RandomForest, CatBoost
  Особенности	Обработка пропусков, категориальные признаки
  Метрики	Accuracy, F1-score
  🛰 Kaggle-челлендж: демонстрация полного ML-конвейера.
10. 🛳 Titanic
  Задача: Классическая бинарная классификация на данных Titanic
  Датасет	Kaggle “Titanic”
  Модели	Logistic Regression, RandomForest
  Особенности	Feature selection, визуализация признаков
  Метрики	Accuracy, F1-score
  🧠 Базовый проект для отработки ML workflow.
11. 🚗 Predicting Road Accident Risk
  Задача: Прогнозирование риска дорожно-транспортных происшествий
  Датасет	Данные о ДТП (время, погода, дорожные условия)
  Модели	LightGBM, XGBoost, CatBoost
  Особенности	Feature engineering (дорожные, погодные факторы)
  Метрики	RMSE
  🛣 Проект ориентирован на анализ влияния погодных и временных факторов на вероятность аварии.
  Включает визуализацию данных, корреляционный анализ и построение предсказательной модели.
🧩 Навыки, отражённые в проектах
  Построение ML и DL моделей с нуля
  Обработка табличных, текстовых и визуальных данных
  Feature engineering и оценка качества моделей
  Оптимизация гиперпараметров (GridSearch, Optuna)
  Визуализация данных и результатов
  Интеграция моделей через FastAPI
  Работа с GitHub и системами контроля версий
