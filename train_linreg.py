import numpy as np
from models.model import linearRegression
from preprocessing import write_csv, get_data
from utils import util

create_text_matrix = True
num_top_words = 100
num_epochs = 10
batch_size = 64
learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1] # hyperparameter
num_occurrences = 10 # hyperparameter


def main():
    # Load data
    train_tweets, train_change = util.get_tweets_change('../train.csv')
    dev_tweets, dev_change = util.get_tweets_change('../dev.csv')
    test_tweets, test_change = util.get_tweets_change('../test.csv')

    # Create dictionary of words that are used at least 'num_occurrences' times
    dictionary = util.create_dict(train_tweets, num_occurrences)
    print('Size of dictionary: ', len(dictionary))

    # Create / load the text matrix
    if create_text_matrix:
        train_matrix = util.transform_text(train_tweets, dictionary)
        write_csv(train_matrix, 'train_matrix.csv')
        dev_matrix = util.transform_text(dev_tweets, dictionary)
        write_csv(dev_matrix, 'dev_matrix.csv')
        test_matrix = util.transform_text(test_tweets, dictionary)
        write_csv(test_matrix, 'test_matrix.csv')
    else:
        train_matrix = np.asarray(get_data('train_matrix.csv'), dtype=float)
        dev_matrix = np.asarray(get_data('dev_matrix.csv'), dtype=float)
        test_matrix = np.asarray(get_data('test_matrix.csv'), dtype=float)

    results = {}

    models, deviation = [], []
    mag_deviation = []
    rmse = []

    # Training loop
    for lr in learning_rates:
        model = linearRegression(learning_rate=lr,
                                 num_epochs=num_epochs,
                                 batch_size=batch_size,
                                 theta=None)
        model.train(train_matrix, train_change, verbose=True)
        results[lr] = model.evaluate(dev_matrix, dev_change)
        """
        if best_deviation is None or results[lr].get('Dev') < best_deviation:
            best_deviation_model = model
            best_deviation = results[lr].get('Dev')
        if best_mag_deviation is None or results[lr].get('Mag_dev') < best_mag_deviation:
            best_mag_deviation_model = model
            best_mag_deviation = results[lr].get('Mag_dev')
        if best_rmse is None or results[lr].get('RMSE') < best_rmse:
            best_rmse_model = model
            best_rmse = results[lr].get('RMSE')
        """
        models.append(model)
        deviation.append(results[lr].get('Dev'))
        mag_deviation.append(results[lr].get('Mag_dev'))
        rmse.append(results[lr].get('RMSE'))
    
    results = []

    # Testing
    for i, model in enumerate(models):
        results.append(model.evaluate(test_matrix, test_change))
        np.savetxt('results_learn_rate=' + str(learning_rates[i]) + '.csv' , results[i]['Preds'])

    top_words = util.get_top_words(num_top_words, train_matrix, train_change, dictionary)
    util.write_json('top_words', top_words)


if __name__ == "__main__":
    main()
