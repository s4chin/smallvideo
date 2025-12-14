wget https://huggingface.co/ramimmo/mini.imgnet.int8/resolve/main/inet.json -O data_cache/inet.json
wget https://huggingface.co/ramimmo/mini.imgnet.int8/resolve/main/inet.npy -O data_cache/inet.npy

python scripts/split_imagenet_train_val.py data_cache/inet.npy data_cache/inet.json

rm data_cache/inet.npy data_cache/inet.json
