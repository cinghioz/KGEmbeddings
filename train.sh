CUDA_VISIBLE_DEVICES=0 python3 -u codes/run.py --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --cpu_num 48 \
 --data_path data/FB15k \
 --model TransE \
 -n 1 -b 150000 -d 512 \
 -g 24.0 -a 1.0 -adv \
 -lr 0.0005 --max_steps 100000 \
 -save models/TransE_FB15k_0 --test_batch_size 256

