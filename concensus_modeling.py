import argparse
import pandas as pd
import build_models as bm
import os
import classic_ml as cml
from build_models import make_dataset
from stats import get_class_stats

AVAILABLE_ALGORITHMS = ['rf', 'bnb', 'knn', 'svm', 'ada']


def score_consensus(data_dir: str,
                    ds: str,
                    endpoint:str,
                    name_col: str,
                    write_dir: str,
                    features=bm.AVAILABLE_DESCRIPTORS,
                    algorithms=AVAILABLE_ALGORITHMS,
                    train_or_ff='five_fold') -> pd.DataFrame:
    """ given a directory, dir, datasets, ds, endpoint, ep, features will score
    the five fold cross validation or training consensus predictions  """
    data_frames = []

    if train_or_ff == 'five_fold':
        train_or_ff = '5fcv'
    else:
        train_or_ff = 'train'

    for feature in features:
        for alg in algorithms:
            filepath = os.path.join(data_dir, 'predictions', f'{alg}_{ds}_{feature}_{endpoint}_None_{train_or_ff}_predictions.csv')
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, index_col=0)

                data_frames.append(df['Binary Prediction'])
            else:
                print("file: {}".format(filepath), " is not found")

    X, y = make_dataset(f'{dataset}.sdf', data_dir=data_dir, features=features[0], name_col=name_col, endpoint=endpoint, threshold=None)

    # take the average acroos all available
    # features spaces and algorithms
    main_frame = pd.concat(data_frames, axis=1)
    consensus = main_frame.mean(1)
    consensus_copy = consensus.copy()
    consensus[consensus_copy < 0.5] = 0
    consensus[consensus_copy >= 0.5] = 1

    consensus.columns = ['Consensus prediction across {len(data_frames)} individual models']
    y = y.loc[consensus.index]

    stats = get_class_stats(None, y, consensus)

    base_file = os.path.join(write_dir, f'{ds}_consensus_{endpoint}_{train_or_ff}_predictions.csv')
    consensus.to_csv(base_file)

    base_file = os.path.join(write_dir, f'{ds}_consensus_{endpoint}_{train_or_ff}_stats.csv')
    pd.DataFrame(stats, index=[0]).to_csv(base_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evalute QSAR Models')
    parser.add_argument('-ds', '--dataset', metavar='ds', type=str, help='training set name')
    parser.add_argument('-f', '--features', metavar='f', type=str, help='features to build model with')
    parser.add_argument('-a', '--algorithms', metavar='a', type=str, help='algorithms to build model with')
    parser.add_argument('-d', '--data_dir', metavar='ev', type=str, help='project directory')
    parser.add_argument('-ep', '--endpoint', metavar='ep', type=str, help='endpoint to model')
    parser.add_argument('-nc', '--name_col', metavar='nc', type=str, help='endpoint to model')
    parser.add_argument('-wd', '--write_dir', metavar='wd', type=str, help='directory to write the consensus model results to')

    args = parser.parse_args()

    dataset = args.dataset
    features = args.features.split(',') if args.features else bm.AVAILABLE_DESCRIPTORS
    algorithms = args.features.split(',') if args.algorithms else AVAILABLE_ALGORITHMS
    name_col = args.name_col
    data_dir = args.data_dir
    endpoint = args.endpoint
    write_dir = args.write_dir

    score_consensus(data_dir, dataset, endpoint, name_col, write_dir, features=features, algorithms=algorithms)


