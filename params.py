
params = dict()

# train params
params['epoch_num'] = 80
params['batch_size'] = 16
params['num_workers'] = 24
params['learning_rate'] = 1e-4
params['step_size'] = 20
params['momentum'] = 0.9
params['weight_decay'] = 1e-5
params['nesterov'] = True
params['display'] = 50
params['use_gpu'] = True
params['gpu_ids'] = [0]

# params['pretrain_model'] = 'result/checkpoint/0418/epoch_8_3.190.cpkt'
params['pretrain_model'] = None

# train log
params['date'] = "0425"
params['train_log'] = "使用新的数据进行训练"

