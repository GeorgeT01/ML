import numpy as np
import matplotlib.pyplot as plt
from mlxtend.regressor import LinearRegression

# 1. Загрузите набор данных ex1data1.txt из текстового файла.
def loadTxtFile(file, column):
    data = np.loadtxt(file, dtype='float', delimiter=',', usecols=(column), unpack=True)
    return data

cities_population = loadTxtFile('data/ex1data1.txt', 0)
profit = loadTxtFile('data/ex1data1.txt', 1)

print('cities population =' , cities_population)
print('profit =' , profit)


# 2. Постройте график зависимости прибыли ресторана от населения города, в котором он расположен.
X = np.array(cities_population)[:, np.newaxis]
y = np.array(profit)

ne_lr = LinearRegression(minibatches=None)
ne_lr.fit(X, y)

print('Intercept: %.2f' % ne_lr.b_)
print('Slope: %.2f' % ne_lr.w_[0])

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')    
    return

lin_regplot(X, y, ne_lr)
plt.show()


# 3. Реализуйте функцию потерь J(θ) для набора данных ex1data1.txt.
def mse(predictions, targets):
    # Получение количества выборок в наборе данных
    samples_num = len(predictions)
    #Суммирование квадратных разностей между прогнозируемыми и ожидаемыми значениями
    accumulated_error = 0.0
    for prediction, target in zip(predictions, targets):
        accumulated_error += (prediction - target)**2
    # Вычисление mean и деление на 2
    mae_error = (1.0 / (2*samples_num)) * accumulated_error
    
    return mae_error


print("MSE = " , mse(ne_lr.b_, ne_lr.w_[0]))


# 4. Реализуйте функцию градиентного спуска для выбора параметров модели. Постройте полученную модель (функцию) совместно с графиком из пункта 2.






