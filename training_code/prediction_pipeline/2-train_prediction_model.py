import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import pickle


def generate_additional_data(samples):

    '''
    :param samples: Number of samples to generate
    :return: Numpy array with the generated data

    This function is build under several assumptions:
    - Predictions that are in the range [0.5,1.0] and are supported by a max in the range [0.5,1.0] might be a 1.0
    - Predictions that are in the range [0.2,0.5] and are supported by a very high max in the range [0.98,1.0] might be a 1.0
    - Predictions that are in the range [0.0,0.5] and the max does not exceed 0.8 might be a 0.0
    - Predictions that are in the range [0.5,0.8] and the max never exceeds 0.5 might be a 0.0

    Note that I mention the max but I only took into account the min and mean which are values with some logic extracted from original data and same logic above

    "This model will allow not only to correct any prediction that was originally wrong, but also to give a more confident prediction"
    '''

    df_new = []
    for i in range(samples//2):
        if i < samples//4:
            rand_pred = np.random.randint(500, 1000) / 1000.
            rand_min = np.random.randint(0, 100) / 1000.
            rand_mean = np.random.randint(100, 500) / 1000.
            rand_max = np.random.randint(500, 1000) / 1000.
            df_new.append([rand_min, rand_mean, rand_max, rand_pred, 1])

        else:
            rand_pred = np.random.randint(200, 500) / 1000.
            rand_min = np.random.randint(200, 700) / 1000.
            rand_mean = np.random.randint(400, 900) / 1000.
            rand_max = np.random.randint(980, 1000) / 1000.
            df_new.append([rand_min, rand_mean, rand_max, rand_pred, 1])

    for i in range(samples//2):
        if i < samples//4:
            rand_pred = np.random.randint(0, 500) / 1000.
            rand_min = np.random.randint(0, 100) / 1000.
            rand_mean = np.random.randint(0, 500) / 1000.
            rand_max = np.random.randint(0, 800) / 1000.
            df_new.append([rand_min, rand_mean, rand_max, rand_pred, 0])

        else:
            rand_pred = np.random.randint(500, 800) / 1000.
            rand_min = np.random.randint(0, 100) / 1000.
            rand_mean = np.random.randint(0, 500) / 1000.
            rand_max = np.random.randint(0, 500) / 1000.
            df_new.append([rand_min, rand_mean, rand_max, rand_pred, 0])

    df_new = np.asarray(df_new)
    return df_new

if __name__ == '__main__':

    # Load training data_other
    df = np.load('training_data.npy')

    # targets are wrong, change them
    for i in range(len(df)):
        if df[i,-1] == 0:
            df[i, -1] = 1
        else:
            df[i, -1] = 0

    # Add low confidence samples with corrective values - THESE ARE VERY IMPORTANT
    df_new = generate_additional_data(samples=400)

    # concatenate both arrays
    df = np.concatenate((df, df_new), axis=0)
    df = shuffle(df)

    x_train, x_test, y_train, y_test = train_test_split(df[:,:-1], df[:,-1], stratify=df[:,-1], test_size=0.2)
    x_train, y_train = shuffle(x_train, y_train)

    # RF classifier
    svm = RandomForestClassifier()
    svm.fit(x_train, y_train)
    score = svm.score(x_test, y_test)
    print(score)

    # Save the model - (It is possible to obtain 100% accurate test score depending on the split)
    filename = 'confidence_final.sav'
    pickle.dump(svm, open(filename, 'wb'))




