import yaml

with open('/data5/sukmin/kaggle/working/gbr.yaml') as f:

    vegetables = yaml.load(f, Loader=yaml.FullLoader)

    print(vegetables)
