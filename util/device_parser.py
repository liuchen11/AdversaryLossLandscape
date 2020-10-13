import os

def config_visible_gpu(device_config):

    if device_config != None and device_config != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = device_config
        print('Use GPU %s'%device_config)
    else:
        print('Use all GPUs' if device_config == None else 'Use CPUs')
