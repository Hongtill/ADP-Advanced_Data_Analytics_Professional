# iris DB 활용 분석

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

iris = load_iris()

iris_data = iris.data

iris_label = iris.target
print('iris target값 : ', iris_label)
print('iris target명 : ', iris.target_names)

iris_df = pd.DataFrame(data=iris.data, columns = iris.feature_names)
iris_df['label'] = iris.target
iris_df.head(3)

X_train,  X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)

# 모델 객체 생성
dt_clf = DecisionTreeClassifier(random_state=11)

# 학습 수행
dt_clf.fit(X_train, y_train)

# 예측
pred = dt_clf.predict(X_test)

from sklearn.metrics import accuracy_score
print("예측 정확도: {0:.4f}".format(accuracy_score(y_test, pred)))
