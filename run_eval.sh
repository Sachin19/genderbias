python -u run.py \
	--mode=eval \
	--data_path=/usr1/home/sachink/data/ethics_project/processed_data_comments_1/chunked/dev_* \
	--eval_data_path=/usr1/home/sachink/data/ethics_project/processed_data_comments_1/chunked/dev_* \
	--vocab_path=/usr1/home/sachink/data/ethics_project/processed_data_comments_1/vocab \
	--log_root=newlog \
	--vocab_size=40000 \
	--hidden_dim=256 \
	--emb_dim=256 \
	--batch_size=128 \
	--encoder_layers=1 \
	--max_enc_steps=200 \
	--adam_lr=0.0001 \
	--optimizer=adam \
	--dropout_input_keep_probability=0.8 \
	--dropout_output_keep_probability=1.0 \
	--exp_name=witheverything2 #>> outfiles/samplemodel 2>&1 \

