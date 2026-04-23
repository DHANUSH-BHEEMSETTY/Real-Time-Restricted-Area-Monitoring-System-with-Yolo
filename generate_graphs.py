import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory
os.makedirs('graphs', exist_ok=True)

# Load data
df = pd.read_csv('data/detection_log.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 1. Bar Chart: Count by Class
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Class', palette='viridis')
plt.title('Total Violations by Class')
plt.xlabel('Object Class')
plt.ylabel('Count')
plt.savefig('graphs/class_violations.png', bbox_inches='tight')
plt.close()

# 2. Histogram: Confidence Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Confidence'], bins=10, kde=True, color='blue')
plt.title('Confidence Distribution of Detections')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.savefig('graphs/confidence_dist.png', bbox_inches='tight')
plt.close()

# 3. Line plot: Detections over Time
plt.figure(figsize=(10, 6))
df.set_index('Timestamp').resample('1min').size().plot()
plt.title('Detections Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Violations')
plt.grid(True)
plt.savefig('graphs/detections_time.png', bbox_inches='tight')
plt.close()

print('Graphs generated successfully in "graphs" directory.')
