import os
import argparse

from azureml.core.run import Run
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

os.makedirs('./outputs', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder')
parser.add_argument('--n-estimators', type=int, default=10, dest='n_estimators', help='n_estimators')
parser.add_argument("--max-depth", type=int, dest="max_depth", default=4)
parser.add_argument("--min-samples-split", type=int, dest="min_samples_split", default=2)
args = parser.parse_args()

print('Data folder is at:', args.data_folder)
print('List all files: ', os.listdir(args.data_folder))

# -----------------------------------------------------------------------------

run = Run.get_context()

n_estimators = args.n_estimators
max_depth = args.max_depth
min_samples_split = args.min_samples_split
run.log('n_estimators', n_estimators)
run.log('max_depth', max_depth)
run.log('min_samples_split', min_samples_split)

df = pd.read_csv(os.path.join(args.data_folder, 'train.csv'))

X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].copy()
y = df['Survived'].copy()

X['Age'] = X['Age'].fillna(X['Age'].median())
assert np.any(X.isna()) == False

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

m = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split)
m.fit(X_train, y_train)

roc_auc = roc_auc_score(y_test, m.predict(X_test))
run.log('roc_auc', roc_auc)

model_file_name = 'rf-{}-{}-{}.pkl'.format(n_estimators, max_depth, min_samples_split)
with open(model_file_name, "wb") as file:
    joblib.dump(value=m, filename='outputs/' + model_file_name)

print('n_estimators={0:.2f}, roc_auc={1:.2f}'.format(n_estimators, roc_auc))
