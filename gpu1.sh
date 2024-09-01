# 消融实验
# test_use_mem
# test_update_mem
# test_loss_type recon corr recon&corr
# async_modeling
# async_type mean max cross_attn line
# for t in True False
# do
#     for u in True False
#     do
#         for y in recon corr reconcorr
#         do 
#             for a in True False
#             do
#                 for p in mean max cross_attn line
#                 do 
#                     python main.py  --dataset SMD --data_path ../Anomaly-Transformer-main/dataset/SMD/ --test_use_mem $t --test_update_mem $u --test_loss_type $y --async_modeling $a --async_type $p --gpu 1 --input_c 38 --output_c 38
#                     python main.py  --dataset MSL --data_path ../Anomaly-Transformer-main/dataset/MSL/ --test_use_mem $t --test_update_mem $u --test_loss_type $y --async_modeling $a --async_type $p --gpu 1 --input_c 55 --output_c 55
#                     python main.py  --dataset SMAP --data_path ../Anomaly-Transformer-main/dataset/SMAP/ --test_use_mem $t --test_update_mem $u --test_loss_type $y --async_modeling $a --async_type $p --gpu 1 --input_c 25 --output_c 25
#                     python main.py  --dataset PSM --data_path ../Anomaly-Transformer-main/dataset/PSM/ --test_use_mem $t --test_update_mem $u --test_loss_type $y --async_modeling $a --async_type $p --gpu 1 --input_c 25 --output_c 25
#                     python main.py  --dataset SWAT --data_path ../TSAD/data/SWAT/A1_A2/ --test_use_mem $t --test_update_mem $u --test_loss_type $y --async_modeling $a --async_type $p --gpu 1 --input_c 50 --output_c 50
#                 done
#             done
#         done
#     done
# done

# 参数敏感实验
# async_gap 20 25 30 35 40 
# for a in 20 25 30 35 40
# do 
#     python main.py  --dataset SMD --data_path ../Anomaly-Transformer-main/dataset/SMD/ --gpu 1 --input_c 38 --output_c 38 --async_gap $a
#     python main.py  --dataset MSL --data_path ../Anomaly-Transformer-main/dataset/MSL/ --gpu 1 --input_c 55 --output_c 55 --async_gap $a
#     python main.py  --dataset SMAP --data_path ../Anomaly-Transformer-main/dataset/SMAP/ --gpu 1 --input_c 25 --output_c 25 --async_gap $a
#     python main.py  --dataset PSM --data_path ../Anomaly-Transformer-main/dataset/PSM/ --gpu 1 --input_c 25 --output_c 25 --async_gap $a
#     python main.py  --dataset SWAT --data_path ../TSAD/data/SWAT/A1_A2/ --gpu 1 --input_c 50 --output_c 50 --async_gap $a
# done
# node_vec_size 1 2 4 6 8 
for n in 1 2 4 6 8
do 
    python main.py  --dataset SMD --data_path ../Anomaly-Transformer-main/dataset/SMD/ --gpu 1 --input_c 38 --output_c 38 --node_vec_size $n
    python main.py  --dataset MSL --data_path ../Anomaly-Transformer-main/dataset/MSL/ --gpu 1 --input_c 55 --output_c 55 --node_vec_size $n
    python main.py  --dataset SMAP --data_path ../Anomaly-Transformer-main/dataset/SMAP/ --gpu 1 --input_c 25 --output_c 25 --node_vec_size $n
    python main.py  --dataset PSM --data_path ../Anomaly-Transformer-main/dataset/PSM/ --gpu 1 --input_c 25 --output_c 25 --node_vec_size $n
    python main.py  --dataset SWAT --data_path ../TSAD/data/SWAT/A1_A2/ --gpu 1 --input_c 50 --output_c 50 --node_vec_size $n
done
# d_model 16 32 64 128
for d in 16 32 64 128
do 
    python main.py  --dataset SMD --data_path ../Anomaly-Transformer-main/dataset/SMD/ --gpu 1 --input_c 38 --output_c 38 --d_model $d
    python main.py  --dataset MSL --data_path ../Anomaly-Transformer-main/dataset/MSL/ --gpu 1 --input_c 55 --output_c 55 --d_model $d
    python main.py  --dataset SMAP --data_path ../Anomaly-Transformer-main/dataset/SMAP/ --gpu 1 --input_c 25 --output_c 25 --d_model $d
    python main.py  --dataset PSM --data_path ../Anomaly-Transformer-main/dataset/PSM/ --gpu 1 --input_c 25 --output_c 25 --d_model $d
    python main.py  --dataset SWAT --data_path ../TSAD/data/SWAT/A1_A2/ --gpu 1 --input_c 50 --output_c 50 --d_model $d
done
