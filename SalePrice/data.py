import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv('C:\\Users\\Master\\Desktop\\ML\\SalePrice\\train.csv')
data_test = pd.read_csv('C:\\Users\\Master\\Desktop\\ML\\SalePrice\\test.csv')
y = data.SalePrice

features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
X = data[features]
X_test = data_test[features]

my_model = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
my_model.fit(X, y)
preds_test = my_model.predict(X_test)
output = pd.DataFrame({'Id': data_test.Id,
                       'SalePrice': preds_test})
output.to_csv('SalePrice\\submission.csv', index=False)

