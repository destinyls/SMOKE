CUDA_VISIBLE_DEVICES=6 python tools/plain_train_net.py --config-file configs/smoke_gn_vector_003nd.yaml
CUDA_VISIBLE_DEVICES=6 python tools/evaluate_script.py --config-file configs/smoke_gn_vector_003nd.yaml

CUDA_VISIBLE_DEVICES=6 python tools/plain_train_net.py --config-file configs/smoke_gn_vector_004nd.yaml
CUDA_VISIBLE_DEVICES=6 python tools/evaluate_script.py --config-file configs/smoke_gn_vector_004nd.yaml