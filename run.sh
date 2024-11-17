export CUDA_VISIBLE_DEVICES=0
python main.py --n_type bw --config config/MECGE_phase.yaml --test
# python main.py --n_type bw --config config/MECGE_complex.yaml
# python main.py --n_type bw --config config/MECGE_wav.yaml
# python test.py --experiments MECGE_phase MECGE_complex