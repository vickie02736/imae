import yaml
config = yaml.load(open('./sw_train_config.yaml', 'r'), Loader=yaml.FullLoader)
a,b,c = config['image_size']
print(a)