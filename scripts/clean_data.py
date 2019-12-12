import numpy as np
import pandas as pd

def str_2_list(data_train, n):
    '''
    Objective:
    Convert the string format into python lists

    Arguments:
    data_train {pandas dataframe} -- contains training data
    n {int} -- column number from data_train to take data from

    Return:
    python list size (num rows, X) where X is number of layers in each
    model form the training set.
    '''

    row = str(data_train.iloc[0,n])
    row = row[1:-1]
    row = row.split(', ')
    row = [float(i) for i in row]

    train_init_params = [row]#np.copy(row)
    num_rows = len(data_train.iloc[:,n])

    for i in range(1, num_rows):
        row = str(data_train.iloc[i,n])
        row = row[1:-1]
        row = row.split(', ')

        try:

            row = [float(j) for j in row]
            train_init_params.append(row)
            #train_init_params = \
            #np.concatenate((train_init_params, np.array(row)), axis=0)
        except:
            print(i, row, '\n')

            #data_train.drop(i)
            #row 1185 is broken dont use

    return train_init_params


def add_initparams_2_dataframe(data, init_params_mu, init_params_std,\
                                                       init_params_l2):
    '''
    Objective:
    Add initParams back to data frame. But each element in initParams
    will become a new column and therefore a feature of the modelself.

    Arguments:
    data {pandas dataframe} -- train or test data

    Return:
    data {pandas dataframe} -- modified
    '''


    init_params_mu_mean  = []
    init_params_std_mean = []
    init_params_l2_mean  = []

    init_params_mu_std  = []
    init_params_std_std = []
    init_params_l2_std  = []

    #add mean and std columns of each inti_param
    for i in range(len(init_params_mu)):

        init_params_mu_mean.append(np.mean(init_params_mu[i][:]))
        init_params_std_mean.append(np.mean(init_params_std[i][:]))
        init_params_l2_mean.append(np.mean(init_params_l2[i][:]))

        init_params_mu_std.append(np.std(init_params_mu[i][:]))
        init_params_std_std.append(np.std(init_params_std[i][:]))
        init_params_l2_std.append(np.std(init_params_l2[i][:]))

    data['init_params_mu_mean']  = init_params_mu_mean
    data['init_params_std_mean'] = init_params_std_mean
    data['init_params_l2_mean']  = init_params_l2_mean

    data['init_params_mu_std']  = init_params_mu_std
    data['init_params_std_std'] = init_params_std_std
    data['init_params_l2_std']  = init_params_l2_std


    #delete original columns
    data = data.drop(columns=['init_params_mu', 'init_params_std',\
                                                'init_params_l2'])
    return data


if __name__ == '__main__':

    train_file = '../data/train.csv'
    test_file  = '../data/test.csv'
    data_train = pd.read_csv(train_file)
    data_test  = pd.read_csv(test_file)

    #print('train.csv: ', list(data_train.keys()))

    #remove useless data
    data_train = data_train.drop(columns=['Unnamed: 0', 'id', 'batch_size_test',\
                             'batch_size_val', 'criterion', 'batch_size_train',\
                             'optimizer'])
    data_test  = data_test.drop(columns=['Unnamed: 0', 'id', 'batch_size_test',\
                             'batch_size_val', 'criterion', 'batch_size_train',\
                             'optimizer'])

    # #drop this broken row
    data_train = data_train.drop(1185)
    #print(list(data_train.iloc[1185,7]))

    #convert init_params to list of lists
    train_init_params_mu  = str_2_list(data_train, 7)
    train_init_params_std = str_2_list(data_train, 8)
    train_init_params_l2  = str_2_list(data_train, 9)

    #test data
    test_init_params_mu  = str_2_list(data_test, 4)
    test_init_params_std = str_2_list(data_test, 5)
    test_init_params_l2  = str_2_list(data_test, 6)

    # print(train_init_params_mu[0][:])

    data_train = add_initparams_2_dataframe(data_train, train_init_params_mu,\
                                                 train_init_params_std,\
                                                 train_init_params_l2)
    data_test = add_initparams_2_dataframe(data_test, test_init_params_mu,\
                                                 test_init_params_std,\
                                                 test_init_params_l2)

    #drop arch_and_hp column
    data_train = data_train.drop(columns=['arch_and_hp'])

    #now get the labels
    data_train_labels = pd.DataFrame()
    data_train_labels['val_error']   = data_train['val_error']
    data_train_labels['train_error'] = data_train['train_error']

    #drop the labels from data_train
    data_train = data_train.drop(columns=['train_error', 'val_error'])

    #drop arch_and_hp column
    data_test = data_test.drop(columns=['arch_and_hp'])

    data_train = data_train.drop(columns=['val_loss', 'train_loss'])

    #normalize columns in data_train(test)
    from sklearn import preprocessing

    x = data_train.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data_train_norm = pd.DataFrame(x_scaled)

    x = data_test.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data_test_norm = pd.DataFrame(x_scaled)

    # print(list(data_test.keys()) == list(data_train.keys()))

    # #save into CSV file
    data_train_norm.to_csv('../data/cleaned_training_data_norm_stats.csv')
    data_train_labels.to_csv('../data/cleaned_training_data_labels_norm_stats.csv')

    data_test_norm.to_csv('../data/cleaned_testing_data_norm_stats.csv')
