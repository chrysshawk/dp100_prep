
# Import libraries
import os
import argparse
from azureml.core import Run, Dataset
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Get script arguments: Dataset ID and test size
parser = argparse.ArgumentParser()
parser.add_argument('--testsplit', type=float, dest='test_split', default=0.3, help='test dataset split')
parser.add_argument('--input-data', type=str, dest='training_dataset_id', help='training dataset')
args = parser.parse_args()

# Set test split
test_split = args.test_split

# Get experiment run context
run = Run.get_context()

# Get training dataset
print('Loading Data...')
student_df = run.input_datasets['training_data'].to_pandas_dataframe()

# create dummies and merge
cat_method = pd.get_dummies(student_df['teaching_method'])
cat_gender = pd.get_dummies(student_df['gender'])
cat_lunch = pd.get_dummies(student_df['lunch'])
cat_school = pd.get_dummies(student_df['school_setting'])
student_df = pd.concat([student_df[['n_student', 'pretest', 'posttest']], 
                        cat_method, cat_gender, cat_lunch, cat_school], axis=1)

# Re-define the target/prediction label
posttest = student_df['posttest']
student_df.drop(columns=['posttest'], inplace=True)
student_df['posttest'] = posttest

# Separate features and labels
X, y = student_df[student_df.columns[0:-1]].values, student_df[student_df.columns[-1]].values

# Split data with input arg 'test_split'
print('Splitting data with train size', str(1-test_split),
      'and test size', str(test_split))
run.log('Test split', np.float(test_split))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_split)

# Train the model in a pipeline

# Define scaling of numeric features
num_features = np.arange(len(X_train[0]))
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# Assign transfomer to preprocessor
preprocessor = ColumnTransformer(transformers=[('num',
                                               num_transformer,
                                               num_features)])

# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor())])

# Fit the pipeline to train
model = pipeline.fit(X_train, (y_train))

# Get predictions
y_hat = model.predict(X_test)

# Generate evaluation metrics
mse = mean_squared_error(y_test, y_hat)
print('MSE:', mse)
run.log('MSE', np.float(mse))

rmse = np.sqrt(mse)
print('RMSE:', rmse)
run.log('RMSE', np.float(rmse))

r2 = r2_score(y_test, y_hat)
print('R2 score:', r2)
run.log('R2 Score', np.float(r2))

# Export the model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/student_scores_model.pkl')

# Complete the run
run.complete()
