CUDA_VISIBLE_DEVICES=7 python tools/plain_train_net.py --config-file configs/smoke_gn_vector.yaml
CUDA_VISIBLE_DEVICES=7 python tools/evaluate_script.py --config-file configs/smoke_gn_vector.yaml

CUDA_VISIBLE_DEVICES=7 python tools/plain_train_net.py --config-file configs/smoke_gn_vector_002nd.yaml
CUDA_VISIBLE_DEVICES=7 python tools/evaluate_script.py --config-file configs/smoke_gn_vector_002nd.yaml