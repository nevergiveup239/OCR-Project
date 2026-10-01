import urllib.request
import yaml

base_url = "https://raw.githubusercontent.com/pbcquoc/vietocr/master/config/base.yml"
vgg_url = "https://raw.githubusercontent.com/pbcquoc/vietocr/master/config/vgg-transformer.yml"

base_config_data = urllib.request.urlopen(base_url).read().decode('utf-8')
vgg_config_data = urllib.request.urlopen(vgg_url).read().decode('utf-8')

base_config = yaml.safe_load(base_config_data)
vgg_config = yaml.safe_load(vgg_config_data)

base_config.update(vgg_config)

with open("vgg_transformer.yml", "w", encoding="utf-8") as f:
    yaml.dump(base_config, f, default_flow_style=False, allow_unicode=True)

print("Downloaded and saved vgg_transformer.yml successfully")
