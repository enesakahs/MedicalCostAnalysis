import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Veri seti yüklenmesi
data = pd.read_csv("Python/AYGAZ MAKINA OGRENMESI/archive/insurance.csv")

#Adım 3
#BMI dağılımının görselleştirilmesi
plt.figure(figsize=(10, 6))
sns.histplot(data['bmi'], bins=30, kde=True)
plt.title('BMI Dağılımı')
plt.show()

#Sigara içenlerin ücretlerini kutu grafiği ile görselleştirilmesi
sns.boxplot(x='smoker', y='charges', data=data)
plt.title('Sigara içen vs Ücretler')
plt.show()

#Bölgelere göre sigara içenlerin dağılımının görselleştirilmesi
sns.countplot(x='region', hue='smoker', data=data)
plt.title('Sigara İçen vs Bölge')
plt.show()


#4 ön isleme
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])
data['smoker'] = label_encoder.fit_transform(data['smoker'])
data = pd.get_dummies(data, columns=['region'], drop_first=True)

#Bağımsız değişkenler (X) ve bağımlı değişken (y) ayırmaca
X = data.drop('charges', axis=1)
y = data['charges']

#Veriyi eğitim ve test kümelerine bölmece
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Veriyi ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#5 model secimi
#Farklı modelleri seç ve çapraz doğrulama ile performansını değerlendir
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor()
]

for model in models: #Her bir model için çapraz doğrulama ile performans değerlendirmesi yapılması
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error') #Negatif ortalama karesel hata kullanarak çapraz doğrulama skorlarını hesapla
    rmse_scores = np.sqrt(-scores) #RMSE skorlarını hesapla
    print(f"{model.__class__.__name__}: Mean RMSE = {rmse_scores.mean()}, Std RMSE = {rmse_scores.std()}")

#6: Hiper-parametre optamizasyon
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestRegressor(random_state=42) #Rastgele orman modelini oluştur
#GridSearchCV ile hiper-parametre optimizasyonu yap
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

best_rf_model = grid_search.best_estimator_

#7 model degerlendirmesi
#test verisi üzerinde tahminler yap ve performansı değerlendir
y_pred = best_rf_model.predict(X_test_scaled) #Test verileri üzerindeki tahminlerin hesaplanması
mse = mean_squared_error(y_test, y_pred) #Gerçek değerler ile tahminler arasındaki kare hataların ortalamasının hesaplanması
rmse = np.sqrt(mse) #MSE'nin karekökünü alarak gerçek değerler ile tahminler arasındaki hataların ortalama büyüklüğünün hesaplanması
mae = mean_absolute_error(y_test, y_pred) #Gerçek değerler ile tahminler arasındaki mutlak hataların ortalamasının hesaplanması

print(f"Ortalama Kare Hatası: {mse}")
print(f"Ortalama Karesel Hata Karekökü: {rmse}")
print(f"Mutlak Hata Ortalaması: {mae}")

