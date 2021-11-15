import os

from argparse import ArgumentParser

import pandas as pd

from rdkit.Chem import SDWriter
from joblib import load

from molecules_and_features import generate_molecules, make_dataset

parser = ArgumentParser(description='Make predictions using trained QSAR models')
parser.add_argument('-ds', '--train_name', metavar='ds', type=str, help='training set name')
parser.add_argument('-ep', '--endpoint', metavar='ep', type=str, help='endpoint to model')
parser.add_argument('-ev', '--env_var', metavar='dd', type=str, help='environmental variable of project directory')
parser.add_argument('-f', '--features', metavar='f', type=str, help='features to build model with')
parser.add_argument('-nc', '--name_col', metavar='nc', type=str, help='name of name column in sdf file')
parser.add_argument('-ps', '--prediction_set', metavar='ps', type=str, help='prediction set name')
parser.add_argument('-t', '--threshold', metavar='t', type=int, help='toxic threshold')
parser.add_argument('-a', '--algorithms', metavar='dir', type=str, help='models to include...should be a csv string')

args = parser.parse_args()
data_dir = os.getenv(args.env_var)
env_var = args.env_var
features_space = args.features.split(',')
name_col = args.name_col
prediction_set = args.prediction_set
endpoint = args.endpoint
threshold = args.threshold
train_name = args.train_name
algorithms = args.algorithms.lower().split(',')

preds = []

if len(algorithms) < 1:
    raise Exception('Please enter at least one algorithm with which to make predictions.')

if len(features_space) < 1:
    raise Exception('Please enter at least one feature space to use to make predictions.')

for features in features_space:
    X_pred = make_dataset(f'{prediction_set}.sdf', data_dir=env_var, features=features, name_col=name_col,
                          endpoint=endpoint,
                          threshold=threshold, pred_set=True)
    for alg in algorithms:

        model_name = f'{alg}_{train_name}_{features}_{endpoint}_{threshold}_pipeline'
        model_file_path = os.path.join(data_dir, 'models', f'{model_name}.pkl')

        if os.path.exists(model_file_path):
            loaded_model = load(model_file_path)
            probabilities = pd.Series(loaded_model.predict_proba(X_pred)[:, 1], index=X_pred.index.astype('str'))
            preds.append(probabilities)

        else:
            raise Exception(f'Model {model_file_path} does not exist.')

if len(preds) > 1:
    consensus_preds = pd.concat(preds, axis=1).mean(1)
    consensus_preds_copy = consensus_preds.copy()
    consensus_preds[consensus_preds_copy >= 0.5] = 1
    consensus_preds[consensus_preds_copy < 0.5] = 0
    final_preds = consensus_preds
else:
    final_preds = preds[0]

molecules = generate_molecules(prediction_set, data_dir)

if len([mol for mol in molecules if mol.HasProp(endpoint)]) > 0:
    X, y = make_dataset(f'{prediction_set}.sdf', data_dir=env_var, features=features, name_col=name_col, endpoint=endpoint,
                        threshold=threshold, cache=False)
    y = y.reindex(X_pred.index)
    y[y.isnull()] = final_preds

    y.to_csv(os.path.join(
        data_dir, 'predictions', f'{prediction_set}_{features}_{endpoint}_{threshold}_no_gaps.csv'),
        header=['Activities'])

    for molecule in molecules:
        if not molecule.HasProp(endpoint):
            molecule.SetProp(endpoint, str(y.loc[molecule.GetProp(name_col)]))

else:
    for molecule in molecules:
        molecule.SetProp(endpoint, str(final_preds.loc[molecule.GetProp(name_col)]))


w = SDWriter(os.path.join(data_dir, f'{prediction_set}_with_predictions.sdf'))

for molecule in molecules:
    w.write(molecule)

w.close()
