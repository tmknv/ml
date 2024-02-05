import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.model_selection import train_test_split


data = pd.read_csv('credit_data.csv', index_col='clientid') //создаем датасет
data=data[data.age > 0] // очищаем от пропусков

sns.scatterplot(data=data,x=data['income'],y=data['loan'],hue=data.default.values) //смотрим на данные

x_train, x_test, y_train, y_test = train_test_split(data[['income','age','loan']], data.default, test_size=.2, random_state=42) //разделяем выборку на то, чем будем тренировать и проверять
precs = []
for k in range(1, 200, 2): //выбираем оптимальное число соседей
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(x_train, y_train)
  precs.append((k, accuracy_score(y_test, knn.predict(x_test))))
#  print(f'{k} : {accuracy_score(y_test, knn.predict(x_test))}')

p = pd.DataFrame(precs, columns=['k', 'accuracy'])
sns.lineplot(p, x='k', y='accuracy') //смотрим 

from sklearn.metrics import confusion_matrix //другой график
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
cm= confusion_matrix(y_test, knn.predict(x_test))
sns.heatmap(cm, annot=True)
