# TransE
CUDA_VISIBLE_DEVICES=0 python3 -u codes/run.py --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --cpu_num 48 \
 --data_path data/umls \
 --model TransE \
 -n 1 -b 1000000 -d 512 \
 -g 24.0 -a 1.0 -adv \
 -lr 0.0005 --max_steps 100000 \
 --test_batch_size 1 \
 -save models/TransE_umls_0 \
 --init_checkpoint models/TransE_umls_0

# # RotatE
#  CUDA_VISIBLE_DEVICES=0 python3 -u codes/run.py --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --cpu_num 48 \
#  --data_path data/FB15k \
#  --model RotatE \
#  -n 1 -b 150000 -d 256 \
#  -g 24.0 -a 1.0 -adv \
#  -lr 0.0005 --max_steps 100000 \
#  -save models/RotatE_FB15k_0 --test_batch_size 128  -de

# # TransR
#   CUDA_VISIBLE_DEVICES=0 python3 -u codes/run.py --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --cpu_num 48 \
#  --data_path data/FB15k \
#  --model TransR \
#  -n 1 -b 50000 -d 512 \
#  -g 24.0 -a 1.0 -adv \
#  -lr 0.0005 --max_steps 100000 \
#  -save models/TransR_FB15k_0 --test_batch_size 64

# DistMult
# CUDA_VISIBLE_DEVICES=0 python3 -u codes/run.py --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --cpu_num 48 \
#  --data_path data/FB15k \
#  --model DistMult \
#  -n 1 -b 500000 -d 512 \
#  -g 100.0 -a 1.0 -adv \
#  -lr 0.0005 --max_steps 100000 \
#  -save models/DistMult_FB15k_0 -r 0.000002 --test_batch_size 128