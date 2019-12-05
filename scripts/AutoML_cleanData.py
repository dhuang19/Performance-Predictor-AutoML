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
            #row 1185 is broken dont use

    return train_init_params

def equal_len_initParams(train_init_params, max_row_size):
    '''
    Objective:
    Make all rows in train_init_params equal length by making them the
    length of the longest row and filling in the remaining entries with
    the mean of the original entries.

    Arguments:
    train_init_params {list of lists} -- contains innit params

    Return:
    train_init_params_new {list of lists} -- all rows are equal length
    '''

    train_init_params_new = []
    for i in range(len(train_init_params)):
        row_size = len(train_init_params[i])
        if row_size != max_row_size:
            new_row  = np.zeros((max_row_size))
            new_row[:row_size] = train_init_params[i]
            new_row[row_size:] = [np.mean(train_init_params[i])] * \
                                 (max_row_size - row_size)

            train_init_params_new.append(new_row)
        else:
            train_init_params_new.append(train_init_params[i])

    return np.array(train_init_params_new)

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

    #add column for each init_param for all rows
    for j in range(len(init_params_mu[0])):
        data['init_params_mu_{}'.format(j)]  = init_params_mu[:,j]
        data['init_params_std_{}'.format(j)] = init_params_std[:,j]
        data['init_params_l2_{}'.format(j)]  = init_params_l2[:,j]

    #delete original columns
    data = data.drop(columns=['init_params_mu', 'init_params_std',\
                                                'init_params_l2'])
    return data

def find_layers(arch_and_hp):
    '''
    Objective:
    Find what layers are used in what order and where the layer is in
    the string

    Arguments:
    arch_and_hp {str} -- contains string arch_and_hp from one row of
    the data

    Return:
    layer_and_position {2D list} -- every contains ->
                         (layer_type, layer_num, start_index, end index)
    '''
    arch_and_hp = '{' + arch_and_hp[12:-2] + '}'

    #possible layer types
    layer_types = ['conv','flatten','tanh','softmax','batchnorm1D','linear',\
                   'dropout','maxpool','leaky_relu','batchnorm','selu','relu']
    layer_and_position = []
    #find the layers and there position in model
    for lt_idx, lt in enumerate(layer_types):
        for j in range(25): #max number of layers in data by inspection
            search = lt+str(j)
            start_idx = arch_and_hp.find(search + ')')

            if start_idx != -1:
                #plus -1,+2 to account for parenthesis
                start_idx -= 1
                end_idx = start_idx + len(search) + 2
                #layer_and_position (layer_type, layer_num, start_index, end index)
                layer_and_position.append([lt_idx, j, start_idx, end_idx])

    #to make all the lists the same length add -1's to data
    num_layers = len(layer_and_position)
    max_layers = 29
    if num_layers < max_layers:
        for i in range(max_layers - num_layers):
            layer_and_position.append([-1,-1,-1,-1])

    layer_and_position = np.array(layer_and_position)

    return layer_and_position


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

    #drop this broken row
    data_train = data_train.drop(1185)

    train_init_params_mu  = str_2_list(data_train, 7)
    train_init_params_std = str_2_list(data_train, 8)
    train_init_params_l2  = str_2_list(data_train, 9)

    # test_init_params_mu  = str_2_list(data_test, 7)
    # test_init_params_std = str_2_list(data_test, 8)
    # test_init_params_l2  = str_2_list(data_test, 9)


    #because layer length is variable so are the init_params length for each
    #model. Therefore we must make them equal length
    #we will add the mean to fill out the vectors to equal length
    #first find the max vector length.
    max=0
    idx=-999
    for i in range(len(train_init_params_std)):
        if max < np.shape(train_init_params_std[i])[0]:
            max = np.shape(train_init_params_std[i])[0]
            idx = i
    #max is 24 for vector length
    #so we will make all the vectors length 24 by filling out the rest with
    #the mean of the previous entries.
    max_row_size = max

    #init_params are the mean, standard deviation and l2 norm of each layer's
    #parameters for every model in the dataset
    train_init_params_mu  = equal_len_initParams(train_init_params_mu, max_row_size)
    train_init_params_std = equal_len_initParams(train_init_params_std, max_row_size)
    train_init_params_l2  = equal_len_initParams(train_init_params_l2, max_row_size)

    #now we can add the data back to the data frame
    data_train = add_initparams_2_dataframe(data_train, train_init_params_mu,\
                                                 train_init_params_std,\
                                                 train_init_params_l2)

    #now we need to convert the arch_and_hp column into a numerical format
    #make into 2D array by formatting the arch_and_hp of all rows
    arch_and_hp = data_train.iloc[0,0]
    layer_and_position = find_layers(arch_and_hp)[:,:2].flatten()

    for i in range(1, len(data_train.iloc[:,0])):
        arch_and_hp = data_train.iloc[i,0]
        layer_and_position_temp = find_layers(arch_and_hp)[:,:2].flatten()
        layer_and_position = np.vstack((layer_and_position,\
                                       layer_and_position_temp))

    #layer num corresponds to layer type
    for i in range(layer_and_position.shape[1]):
        data_train['layer_type_{}'.format(i)] = layer_and_position[:,i]
        data_train['layer_num_{}'.format(i)]  = layer_and_position[:,i]

    #drop arch_and_hp column
    data_train = data_train.drop(columns=['arch_and_hp'])

    #now get the labels
    data_train_labels = pd.DataFrame()
    data_train_labels['train_error'] = data_train['train_error']
    data_train_labels['val_error']   = data_train['val_error']

    #drop the labels from data_train
    data_train = data_train.drop(columns=['train_error', 'val_error'])

    #save into CSV file
    data_train.to_csv('../data/cleaned_training_data.csv')
    data_train_labels.to_csv('../data/cleaned_training_data_labels.csv')

    #print(data_train_labels.iloc[0:6,:])
    #print(list(data_train.iloc[i,:]))
    # print(np.shape(data_train))
    # print(list(data_train.keys()))
    #
    # for j in range(len(data_train.iloc[0,:])):
    #     try:
    #         len(data_train.iloc[0,j])
    #         print(data_train.iloc[0,j])
    #     except:
    #         print('nice')
