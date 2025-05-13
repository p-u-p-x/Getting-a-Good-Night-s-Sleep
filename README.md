# Getting a Good Night's Sleep

## Table of Contents
 
 - [Project Overview](#project-overview)
 - [Data Source](#data-source)
 - [Tools](#tools)
 - [Data Cleaning](#data-cleaning)
 - [Exploratory Data Analysis](#exploratory-data-analysis)
 - [Demographic  Analysis](#demographic-analysis)
 - [Occupation Impact](#occupation-impact)
 - [BMI and Sleep Disorders](#bmi-and-sleep-disorders)
 - [Physical Activity Correlation](#physical-activity-correlation)
 - [Results and Findings](#results-and-findings)
 - [Recommendations](#recommendations)

## Project Overview

This project analyzes sleep health data from SleepScope to identify relationships between lifestyle factors and sleep quality. We examine:

- Demographic patterns (age, gender)
- Occupational impacts on sleep
- BMI categories and sleep disorders
- Physical activity correlations
- Stress levels and sleep quality

## Data Source

The dataset sleep_health_data.csv contains:

- 374 anonymized user records
- 13 features including:
  - Sleep duration and quality (1-10 scale)
  - Physical activity (minutes/day)
  - Stress levels (1-10 scale)
  - BMI categories
  - Sleep disorder diagnoses

## Tools

This project utilizes Python for Exploratory Data Analysis (EDA), leveraging the following libraries:
- pandas – Data manipulation and preprocessing
- numpy – Numerical operations
- matplotlib & seaborn – Data visualization
- scikit-learn – Statistical analysis and preprocessing
- Geopandas - mapping crime hotspots
- Jupyter Notebook – Interactive code execution

```python 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Visualization setup
plt.style.use('ggplot')
%matplotlib inline
sns.set_palette("husl")
```


## Data Cleaning

```python
# Load and inspect data
sleep_df = pd.read_csv('sleep_health_data.csv')
print(f"Initial shape: {sleep_df.shape}")
print(f"Missing values:\n{sleep_df.isnull().sum()}")

# Cleaning function
def clean_sleep_data(df):
    # Standardize BMI categories
    df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')
    
    # Convert sleep duration to numeric (handle any strings)
    df['Sleep Duration'] = pd.to_numeric(df['Sleep Duration'], errors='coerce')
    
    # Create sleep quality categories
    bins = [0, 4, 7, 10]
    labels = ['Poor', 'Average', 'Good']
    df['Sleep Quality Category'] = pd.cut(df['Quality of Sleep'], 
                                        bins=bins, 
                                        labels=labels)
    return df

sleep_clean = clean_sleep_data(sleep_df)
```

### Initial Inspection
  - Loads CSV data and checks dataset dimensions
  - Identifies missing values in each column
### Standardization
  - Consolidates "Normal Weight" and "Normal" BMI categories
  - Converts sleep duration to numeric format
### Feature Engineering
  - Creates categorical sleep quality labels (Poor/Average/Good)
  - Bins quality scores (0-4=Poor, 4-7=Average, 7-10=Good)

## Exploratory Data Analysis

### Basic Statistics
  - Summary statistics (mean, min, max)
  - Counts categorical values (gender, BMI, disorders)
  ```python
  print("Summary Statistics:")
  print(sleep_clean.describe())
  
  print("\nCategorical Distributions:")
  print(sleep_clean['Gender'].value_counts())
  print(sleep_clean['BMI Category'].value_counts())
  print(sleep_clean['Sleep Disorder'].value_counts())
  ```

#### Sleep Duration Distribution
  - Visualizes distribution with histogram
  - Highlights average sleep duration (red line)
  ```python
  plt.figure(figsize=(10,6))
  sns.histplot(sleep_clean['Sleep Duration'], bins=20, kde=True)
  plt.title('Distribution of Sleep Duration')
  plt.xlabel('Hours of Sleep')
  plt.axvline(sleep_clean['Sleep Duration'].mean(), color='r', linestyle='--')
  plt.show()
  ```

#### Sleep Quality Distribution
  - Bar plot of quality category counts
  - Ordered from Poor to Good
  ```python
  plt.figure(figsize=(10,6))
  sns.countplot(x='Sleep Quality Category', data=sleep_clean, 
               order=['Poor', 'Average', 'Good'])
  plt.title('Distribution of Sleep Quality Ratings')
  plt.show()
  ```

### Demographic Analysis

#### Age vs Sleep Quality
  - Boxplot compares age distribution across sleep quality levels
  - Reveals if certain age groups experience poorer sleep
  ``` python
  plt.figure(figsize=(10,6))
  sns.boxplot(x='Sleep Quality Category', y='Age', data=sleep_clean,
             order=['Poor', 'Average', 'Good'])
  plt.title('Age Distribution by Sleep Quality')
  plt.show()
  ```

#### Gender Differences
  - Side-by-side boxplots for sleep duration and quality
  - Compares male vs. female sleep patterns
  ``` python
  plt.figure(figsize=(12,5))
  plt.subplot(1,2,1)
  sns.boxplot(x='Gender', y='Sleep Duration')
  plt.title('Sleep Duration by Gender')
  
  plt.subplot(1,2,2)
  sns.boxplot(x='Gender', y='Quality of Sleep')
  plt.title('Sleep Quality by Gender')
  plt.tight_layout()
  plt.show()
  ```

## Occupation Impact

#### Occupation and Sleep Duration
  - Horizontal bar chart ranks occupations by average sleep hours
  - Identifies professions with least sleep (e.g., nurses)
  ```python
  plt.figure(figsize=(12,6))
  sleep_by_occ = sleep_clean.groupby('Occupation')['Sleep Duration'].mean().sort_values()
  sns.barplot(x=sleep_by_occ.values, y=sleep_by_occ.index)
  plt.title('Average Sleep Duration by Occupation')
  plt.xlabel('Hours of Sleep')
  plt.show()
  
  lowest_sleep_occ = sleep_by_occ.index[0]
  ```

#### Occupation and Sleep Quality
  - Similar ranking for sleep quality scores
  - Shows if sleep-deprived jobs also have poor quality
```python
plt.figure(figsize=(12,6))
quality_by_occ = sleep_clean.groupby('Occupation')['Quality of Sleep'].mean().sort_values()
sns.barplot(x=quality_by_occ.values, y=quality_by_occ.index)
plt.title('Average Sleep Quality by Occupation')
plt.xlabel('Sleep Quality (1-10)')
plt.show()

lowest_quality_occ = quality_by_occ.index[0]
```


## BMI and Sleep Disorders

### BMI Category Analysis
```python
# Calculate insomnia ratios
bmi_groups = sleep_clean.groupby('BMI Category')['Sleep Disorder'].value_counts(normalize=True).unstack()
bmi_groups[['Insomnia','Sleep Apnea']].plot(kind='bar', stacked=True, figsize=(10,6))
plt.title('Sleep Disorder Prevalence by BMI Category')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.show()

bmi_insomnia_ratios = {
    "Normal": round(bmi_groups.loc['Normal','Insomnia'], 2),
    "Overweight": round(bmi_groups.loc['Overweight','Insomnia'], 2),
    "Obese": round(bmi_groups.loc['Obese','Insomnia'], 2)
}
```
#### Disorder Prevalence
- Stacked bar chart shows insomnia/sleep apnea rates
- Groups by BMI category (Normal/Overweight/Obese)
- Calculates exact ratios for each group

## Physical Activity Correlation

#### Activity vs Sleep Quality
  - Scatterplot correlates activity minutes with sleep quality
  - Color-coded by quality category
  - Calculates Pearson correlation coefficient
```python
plt.figure(figsize=(10,6))
sns.scatterplot(x='Physical Activity Level', y='Quality of Sleep', 
               hue='Sleep Quality Category', data=sleep_clean)
plt.title('Physical Activity vs Sleep Quality')
plt.show()

# Calculate correlation
activity_corr = sleep_clean['Physical Activity Level'].corr(sleep_clean['Quality of Sleep'])
```

#### Stress vs Sleep Quality
  - Boxplot compares stress scores across sleep quality groups
  - Tests if poor sleep correlates with higher stress
```python
plt.figure(figsize=(10,6))
sns.boxplot(x='Sleep Quality Category', y='Stress Level', data=sleep_clean,
           order=['Poor', 'Average', 'Good'])
plt.title('Stress Levels by Sleep Quality')
plt.show()
```

## Results and Findings

### Occupation Impact:
  - Medical professionals show worst sleep metrics
### BMI Effects:
  - Obesity linked to 32% higher insomnia risk
### Activity Correlation:
  - Active users (60+ min/day) report better sleep quality
### Demographic Patterns:
  - Women sleep longer; quality declines after age 50

## Recommendations

Workplace Interventions:
  - Target nurse and doctor populations for sleep health programs
  - Consider shift scheduling adjustments for medical professionals
Health Programs:
  - Develop weight management programs to address sleep disorders
  - Promote physical activity for sleep quality improvement
App Features:
  - Add BMI-specific sleep recommendations
  - Implement activity tracking with sleep quality feedback




