import torch
import os
import pandas as pd

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)

def init_ortho(m):
    """ Initializes weight layers with orthonormal matrix
    Use with network.apply(init_ortho)
    """ 
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight)

def get_n_params(model):
    """ Returns the number of learnable parameters of a network
    """
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def add_noise(inputs):
    """ Adds noise to a tensor
    makes sure it stays between 0 and 1
    """
    noise = torch.clip(torch.randn_like(inputs)*0.01, min=0, max=1)
    return inputs + noise


def accuracy(y_pred, y_true):
    """ Computes the accuracy between 2 class tensors
    """
    y_pred = torch.round(y_pred)
    y_true = torch.round(y_true)
    right = (y_pred == y_true)
    return (torch.sum(right) / len(right)).item()


def minmax_scale(v, new_min, new_max):
    """ Scales tensor between new_min and new_max

    Args:
        v : tensor to scale
        new_min : minimum value in the tensor
        new_max : maximum value in the tensor
    """
    with torch.no_grad():
        v_min, v_max = v.min(), v.max()
        v = (v - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
    return v


def apply_weight_decay(*modules, weight_decay_factor=0., wo_bn=True):
    '''
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/5
    Apply weight decay to pytorch model without BN;
    In pytorch:
        if group['weight_decay'] != 0:
            grad = grad.add(p, alpha=group['weight_decay'])
    p is the param;
    :param modules:
    :param weight_decay_factor:
    :return:
    '''
    for module in modules:
        for m in module.modules():
            if hasattr(m, 'weight'):
                if wo_bn and isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    continue
                if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
                    m.weight.grad += m.weight * weight_decay_factor

    

def write_params(p, folder='saved_models', verbose=0):
    """ Writes the params in a separate parameter file
    """
    filename = p['filename']

    string = f"""Name : {p['filename']}
Last epoch : {p['epoch']}
########### GLOBAL ###########
DS: {p['ds']}
Run test : {p['run_test']}
Batch size : {p['bs']}
Crop_size : {p['crop_size']}\n\n"""

    string += f"""########### ARCHI ###########
Input dim : {p['z_dim']}
{p['archi_info']}\n\n"""

    string += f"""########### TRAINING PARAMS ###########\n
Epochs : {p['n_epoch']}
Save freq : {p['save_frequency']}
Discriminator learning factor (k) : {p['k']}\n\n"""

    string += f"""########### MODEL PARAMS ###########
lrG : {p['lrG']}
lrD : {p['lrD']}
beta : {p['beta1']}
Weight decay Discriminator : {p['weight_decayD']}
Weight decay Generator : {p['weight_decayG']}
label_reals : {p['label_reals']}
label_fakes :{p['label_fakes']}"""

    if verbose:
        print(string)
        print()
        print("#######################\n")

    filename += "-PARAMS"
    with open(os.path.join(folder, filename), 'w+') as file :
        file.write(string)
        file.close()

def get_epoch_from_log(param_dict, folder='saved_models', verbose=1):
    """ Reads the PARAMS file to fill the param dict with
    the correct parameters
    """
    with open(os.path.join(folder, param_dict["filename"] + "-PARAMS"), "r") as f:
        lines = pd.Series(f.readlines()).str.strip('\n')
    
    #### Verif paramètres égaux TODO ###

    search = {'filename': 'Name', 'archi_info': 'upsample type',
        'lrG':'lrG', 'lrD': 'lrD', 'beta1': 'beta', 
        'weight_decayD': 'Weight decay Discriminator',
        'weight_decayG': 'Weight decay Generator',
        'k': "Discriminator learning factor (k) : 2",
        'z_dim': 'Input dim', 'n_epoch': 'Epochs',
        'save_frequency': 'Save freq', 'label_fakes': 'label_fakes',
        'label_reals': 'label_reals', 'ds': 'DS', 'run_test': "Run test",
        'bs': "Batch size", 'crop_size': "Crop_size", 'epoch':"Last epoch"}
    for param in param_dict.keys():
        try:
            line = lines[lines.str.startswith(search[param])]
            pp = line.values[0].split(":")[1].strip()
            if param not in ['filename', 'archi_info', 'ds', 'run_test']:
                pp = float(pp)
                if int(pp) == pp:
                    pp = int(pp)
            param_dict[param] = pp
        except Exception as e:
            print(f"Could not retrieve {param} from log : {e}")
