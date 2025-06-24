# Titanic Dataset Preprocessing

This project performs data preprocessing on the Titanic dataset using Python and libraries like Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

## 📦 Dataset

- File: `Titanic-Dataset.csv`
- Source: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data) (or a manually created dataset)
- Contains information about passengers, such as age, class, fare, gender, and survival status.

## ✅ Steps Performed

1. **Import Libraries & Dataset**
   - Load the CSV file into a Pandas DataFrame.

2. **Explore Dataset**
   - Display basic info, check for missing values and data types.

3. **Handle Missing Values**
   - `Age`: Filled using **median**.
   - `Embarked`: Filled using **mode**.
   - `Cabin`: Dropped (too many missing values).

4. **Encode Categorical Features**
   - `Sex`: Label encoding (`male` → 0, `female` → 1).
   - `Embarked`: One-hot encoded.

5. **Normalize Numerical Features**
   - Standardized `Age`, `Fare`, `SibSp`, `Parch` using `StandardScaler`.

6. **Outlier Detection & Removal**
   - Used **IQR method** and **boxplots** to detect and remove outliers from numerical columns.

## 📊 Visualization

- Boxplots are used to visually inspect the spread of numerical features and detect outliers.

## 🧪 Libraries Required

Install them with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
