import ast
import fnmatch
import re

arch_and_hp = 'Sequential( (conv0): Conv2d(3, 28, kernel_size=(1, 1), stride=(1, 1)) (flatten1): Flatten() (linear2): Linear(in_features=28672, out_features=29, bias=True) (batchnorm1D3): BatchNorm1d(29, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) (tanh4): Tanh() (tanh5): Tanh() (tanh6): Tanh() (dropout7): Dropout(p=0.714631919301416) (linear8): Linear(in_features=29, out_features=47, bias=True) (dropout9): Dropout(p=0.97788680890307) (linear10): Linear(in_features=47, out_features=10, bias=True) (tanh11): Tanh() (softmax12): Softmax() )'

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

            #print(arch_and_hp[start_idx: end_idx])
print(layer_and_position)
# arch_and_hp = arch_and_hp.split(' |{|}')


# print(arch_and_hp)
# arch_and_hp = ast.literal_eval(arch_and_hp)
# print(arch_and_hp)







 # Sequential( (conv0): Conv2d(3, 28, kernel_size=(1, 1), stride=(1, 1))
 #  (flatten1): Flatten()
 #  (linear2): Linear(in_features=28672, out_features=29, bias=True)
 #  (batchnorm1D3): BatchNorm1d(29, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
 #  (tanh4): Tanh()
 #  (tanh5): Tanh()
 #  (tanh6): Tanh()
 #  (dropout7): Dropout(p=0.714631919301416)
 #  (linear8): Linear(in_features=29, out_features=47, bias=True)
 #  (dropout9): Dropout(p=0.97788680890307)
 #  (linear10): Linear(in_features=47, out_features=10, bias=True)
 #  (tanh11): Tanh()
 #  (softmax12): Softmax() )
