import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

df_train = pd.read_csv(r"Data\train.csv")
df_test = pd.read_csv(r"Data\test.csv")

print(df_train.head())
print(df_test.head())

df_train.info()
df_test.info()

df_train.isnull().sum()
df_test.isnull().sum()

df_train.describe()
df_test.describe()

Q1 = df_train['Price'].quantile(0.25)
Q3 = df_train['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df_train[(df_train['Price'] <= lower_bound) | (df_train['Price'] >= upper_bound)]
df_train.drop(outliers.index, inplace=True)
df_train.shape

Q1 = df_test['Weight Capacity (kg)'].quantile(0.25)
Q3 = df_test['Weight Capacity (kg)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df_test[(df_test['Weight Capacity (kg)'] <= lower_bound) | (df_test['Weight Capacity (kg)'] >= upper_bound)]
df_test.drop(outliers.index, inplace=True)
df_test.shape

df_train.drop_duplicates()
df_test.drop_duplicates()

df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

unique_values = ['Brand' ,'Material','Size','Compartments','Laptop Compartment' , 'Waterproof' ,'Style' ,'Color','Weight Capacity (kg)','Price']
for col in unique_values:
    print(f"Unique values in column '{col}' :")
    print(df_train[col].unique())

kategori_kolom = ['Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# Loop untuk menghitung rata-rata harga berdasarkan setiap kategori
for col in kategori_kolom:
    harga_rata2 = df_train.groupby(['Brand',col])['Price'].mean()  # Groupby per kategori
    
    # Print hasil
    print(f"\nHarga rata-rata berdasarkan {col}:")
    for kategori, avg_price in harga_rata2.items():
        print(f"  {kategori}: {avg_price:.2f}")
harga_ratarata_material = df_train.groupby('Material')['Price'].mean()
for material , avg_price_material in harga_ratarata_material.items():
    print(f'Harga rata rata material {material} adalah {avg_price_material}')
harga_ratarata_size = df_train.groupby('Size')['Price'].mean()
for size , avg_price_size in harga_ratarata_size.items():
    print(f'Harga rata rata size {size} adalah {avg_price_size}')
harga_ratarata_compartments = df_train.groupby('Compartments')['Price'].mean()
for compartments , avg_price_compartments in harga_ratarata_compartments.items():
    print(f'Harga rata rata compartments {compartments} adalah {avg_price_compartments}')
harga_ratarata_Laptop_Compartment = df_train.groupby('Laptop Compartment')['Price'].mean()
for Laptop_Compartment , avg_price_Laptop_Compartment in harga_ratarata_Laptop_Compartment.items():
    print(f'Harga rata rata Laptop Compartment {Laptop_Compartment} adalah {avg_price_Laptop_Compartment}')
harga_ratarata_waterproof = df_train.groupby('Waterproof')['Price'].mean()
for waterproof , avg_price_waterproof in harga_ratarata_waterproof.items():
    print(f'Harga rata rata waterproof {waterproof} adalah {avg_price_waterproof}')
harga_ratarata_style = df_train.groupby('Style')['Price'].mean()
for style , avg_price_style in harga_ratarata_style.items():
    print(f'Harga rata rata style {style} adalah {avg_price_style}')
harga_ratarata_color = df_train.groupby('Color')['Price'].mean()
for color , avg_price_color in harga_ratarata_color.items():
    print(f'Harga rata rata color {color} adalah {avg_price_color}')
harga_ratarata_Brand = df_train.groupby('Brand')['Price'].mean()
for Brand , avg_price_Brand in harga_ratarata_Brand.items():
    print(f'Harga rata rata Brand {Brand} adalah {avg_price_Brand}')
filtered_df = df_train[
    (df_train['Brand'] == 'Jansport') & 
    (df_train['Material'] == 'Canvas') & 
    (df_train['Size'] == 'Large') & 
    (df_train['Compartments'] == 8.0) &  # Pastikan ini tipe numerik
    (df_train['Laptop Compartment'] == 'Yes') & 
    (df_train['Waterproof'] == 'No') & 
    (df_train['Style'] == 'Tote') & 
    (df_train['Color'] == 'Green')
]

# Hitung rata-rata harga dari hasil filter
average_price = filtered_df['Price'].mean()

print(f"Harga rata-rata untuk backpack dengan spesifikasi yang dipilih adalah: {average_price:.2f}")
filtered_df_min = df_train[
    (df_train['Brand'] == 'Adidas') & 
    (df_train['Material'] == 'Leather') & 
    (df_train['Size'] == 'Medium') & 
    (df_train['Compartments'] == 9.0) &  # Pastikan ini tipe numerik
    (df_train['Laptop Compartment'] == 'No') & 
    (df_train['Waterproof'] == 'Yes') & 
    (df_train['Style'] == 'Messenger') & 
    (df_train['Color'] == 'Black')
]

# Hitung rata-rata harga dari hasil filter
average_price = filtered_df_min['Price'].mean()

print(f"Harga rata-rata untuk backpack dengan spesifikasi yang dipilih adalah: {average_price:.2f}")
df_train.hist(bins=50, figsize=(20,15))
plt.show()
Backpack_distribution = ['Brand' ,'Material','Size','Compartments','Laptop Compartment' , 'Waterproof' ,'Style' ,'Color']
for col in Backpack_distribution:
    plt.figure(figsize=(8,4))
    sns.countplot(x=df_train[col] , data=df_train , palette='viridis')
    plt.title(f'Distribusi{col}')
    plt.xticks(rotation=45)
    plt.show()
import scipy.stats as stats
import pandas as pd


categorical_features = ['Brand' ,'Material','Size','Compartments','Laptop Compartment' , 'Waterproof' ,'Style' ,'Color','Price']

# Membuat dataframe untuk menyimpan hasil uji chi-square
chi2_results = pd.DataFrame(columns=['Feature_1', 'Feature_2', 'Chi2', 'p-value'])

# Loop untuk menguji setiap pasangan fitur kategorikal
for i in range(len(categorical_features)):
    for j in range(i+1, len(categorical_features)):  # Agar tidak menguji fitur yang sama dua kali
        feature1 = categorical_features[i]
        feature2 = categorical_features[j]
        
        # Membuat tabel kontingensi
        contingency_table = pd.crosstab(df_train[feature1], df_train[feature2])
        
        # Melakukan uji Chi-Square
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Menyimpan hasil
        chi2_results = pd.concat([chi2_results, 
                                  pd.DataFrame({'Feature_1': [feature1], 'Feature_2': [feature2], 
                                                'Chi2': [chi2], 'p-value': [p]})], ignore_index=True)

# Menampilkan hasil
chi2_results.sort_values(by='p-value')

# Pisahkan variabel independen (X) dan dependen (y)
X = df_train.drop('Price', axis=1)
y = df_train['Price']

# Encode variabel kategorikal menggunakan OneHotEncoder
categorical_features = ['Color', 'Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof']
numeric_features = ['Compartments' ,'Weight Capacity (kg)']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', 'passthrough', numeric_features)
    ])

# Buat pipeline: Preprocessing -> Polynomial Features -> Linear Regression
model = Pipeline([
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2)),  # Derajat polinomial (bisa disesuaikan)
    ('regressor', LinearRegression())
])

# Bagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}')
print(f'R2 Score: {r2}')

predicted_price = model.predict(df_test )
df_result = pd.DataFrame({
    'id': df_test['id'],
    'Predicted Price': predicted_price
})
df_result

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'{model_name} Metrics:')
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'R2 Score: {r2}')
    print('-' * 50)
    return {'Model': model_name, 'MSE': mse, 'MAE': mae, 'R2': r2}

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regression': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regression': GradientBoostingRegressor(random_state=42)
}

# Hasil evaluasi akan disimpan di sini
results = []

# Latih dan evaluasi setiap model
for model_name, model in models.items():
    # Buat pipeline: Preprocessing -> Model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Latih model
    pipeline.fit(X_train, y_train)
    
    # Prediksi
    y_pred = pipeline.predict(X_test)
    
    # Evaluasi model
    metrics = evaluate_model(y_test, y_pred, model_name)
    results.append(metrics)

# Tampilkan hasil evaluasi
results_df = pd.DataFrame(results)
print(results_df)

best_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])
best_model.fit(X_train, y_train)
predicted_price = best_model.predict(df_test)
df_result = pd.DataFrame({
    'id': df_test['id'],
    'Predicted Price': predicted_price
})
df_result

label_encoders = {}
for column in df_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_train[column] = le.fit_transform(df_train[column])
    label_encoders[column] = le 

df_train.head() 

X = df_train.drop('Price',axis=1)
y = df_train['Price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert X to a Pandas DataFrame if it is a NumPy array
X = pd.DataFrame(X)

# Identify the categorical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Create the preprocessor with improvements
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), X.select_dtypes(exclude=['object']).columns.tolist()),
        ('cat', OneHotEncoder(sparse_output=True, handle_unknown='ignore'), categorical_features),  # Changed sparse_output and handle_unknown
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Fit and transform the data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

import tensorflow.keras.backend as K

# Definisikan metrik R-squared (R2)
def r2_metric(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - (ss_res / (ss_tot + K.epsilon()))

def create_model(input_dim=78, dropout_rate=0.2):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(1, activation='linear')  
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',  
        metrics=['mae', r2_metric]  # Tambahkan metrik R2
    )

    return model

# Buat model
model = create_model(input_dim=X_train.shape[1])

# Latih model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=128,
    validation_data=(X_test, y_test),
    verbose=1
)

eval_result = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Loss (MSE): {eval_result[0]:.4f}")
print(f"Final Test MAE: {eval_result[1]:.4f}")

label_encoders = {}
for column in df_test.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_test[column] = le.fit_transform(df_test[column])
    label_encoders[column] = le 

df_test.head() 

# Pastikan df_test telah diproses seperti X_train sebelum prediksi
df_test_processed = preprocessor.transform(df_test)  
# Lakukan prediksi
predicted_price = model.predict(df_test_processed).flatten()  # Ubah ke bentuk 1D

# Buat DataFrame hasil prediksi
df_result = pd.DataFrame({
    'id': df_test['id'],  # Pastikan kolom 'id' ada dalam df_test
    'Predicted Price': predicted_price
})

# Tampilkan hasil
df_result
