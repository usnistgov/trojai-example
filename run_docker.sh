docker run --gpus 0 -it \
-v /home/carnejk1/mitigation_data:/trojai-example/data/:ro \
-v /home/carnejk1/trojai-example-mitigation/out:/trojai-example/output/:rw \
mitigation-test \
--mitigate \
--model_filepath /trojai-example/data/final_model_state_dict.pt \
--clean_trainset /trojai-example/data/dataset/clean_trainset \
--poison_trainset /trojai-example/data/dataset/poison_trainset \
--output_dirpath /trojai-example/output \
--num_classes 10