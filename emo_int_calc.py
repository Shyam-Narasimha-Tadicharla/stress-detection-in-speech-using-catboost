import pandas as pd

# Load the dataset
data = pd.read_csv('./tabulatedVotes.csv')

data['is_stress'] = data['emoVote'].apply(lambda x: 1 if x in ['A', 'F', 'S'] else 0)

from sklearn.model_selection import train_test_split

X = data[['meanAngerResp', 'meanDisgustResp', 'meanFearResp', 
          'meanHappyResp', 'meanNeutralResp', 'meanSadResp']]
y = data['is_stress']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

from sklearn.metrics import classification_report

y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Non-Stress', 'Stress']))

feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importance)

# # Calculate frequency of each emotion
# emotion_columns = ['A', 'D', 'F', 'H', 'N', 'S']
# emotion_frequencies = data[emotion_columns].sum()

# # Display the frequencies
# print(emotion_frequencies)

# from sklearn.decomposition import FactorAnalysis
# from sklearn.preprocessing import StandardScaler

# # Select relevant columns for factor analysis
# response_columns = ['meanAngerResp', 'meanDisgustResp', 'meanFearResp', 'meanHappyResp', 'meanNeutralResp', 'meanSadResp']
# X = data[response_columns]

# # Standardize the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Perform Factor Analysis
# fa = FactorAnalysis(n_components=3)  # Adjust number of components based on your analysis
# X_factors = fa.fit_transform(X_scaled)

# # Display factor loadings
# print(fa.components_)

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt

# # Define target and features
# X = data[response_columns]
# y = data['stress_level']  # Replace with actual stress level column

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train Random Forest
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)

# # Get feature importances
# importances = rf.feature_importances_

# # Plot feature importances
# plt.barh(response_columns, importances)
# plt.xlabel('Feature Importance')
# plt.title('Feature Importance in Predicting Stress')
# plt.show()