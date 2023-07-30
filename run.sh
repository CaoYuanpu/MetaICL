
python3 test.py --dataset ag_news --k 16 --split test --seed 100 --use_demonstrations --test_batch_size 16 --method channel --checkpoint checkpoints/channel-metaicl/class_to_class/model.pt --out_dir checkpoints/channel-metaicl/class_to_class --log_file log/ag_news_seed_100_channel.log

python3 test.py --dataset ag_news --k 16 --split test --seed 100 --use_demonstrations --test_batch_size 16 --method direct --checkpoint checkpoints/metaicl/class_to_class/model.pt --out_dir checkpoints/metaicl/class_to_class --log_file log/ag_news_seed_100_direct.log
