
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Path to the tabulatedVotes.csv file
file_path = './tabulatedVotes.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Select the normalized mean emotion columns
emotion_columns = ['meanAngerRespNorm', 'meanDisgustRespNorm', 'meanFearRespNorm', 
                   'meanHappyRespNorm', 'meanNeutralRespNorm', 'meanSadRespNorm']

# Calculate the Pearson correlation matrix
correlation_matrix = df[emotion_columns].corr(method='pearson')

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Pearson Correlation Heatmap of Emotions in CREMA-D Dataset')

# Rename the axis labels
emotion_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
plt.xticks(range(len(emotion_names)), emotion_names, rotation=45)
plt.yticks(range(len(emotion_names)), emotion_names, rotation=45)

plt.tight_layout()
plt.show()

# Print the correlation matrix
print(correlation_matrix)


# Calculate the Spearman correlation matrix
correlation_matrix_p = df[emotion_columns].corr(method='spearman')

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_p, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Spearman Correlation Heatmap of Emotions in CREMA-D Dataset')

# Rename the axis labels
emotion_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
plt.xticks(range(len(emotion_names)), emotion_names, rotation=45)
plt.yticks(range(len(emotion_names)), emotion_names, rotation=45)

plt.tight_layout()
plt.show()

# Print the correlation matrix
print(correlation_matrix_p)