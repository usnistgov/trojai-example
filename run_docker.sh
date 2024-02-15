docker run --gpus 0 -it \
-v /home/carnejk1/mitigation_data:/trojai-example/data/:ro \
-v /home/carnejk1/trojai-example-mitigation/out:/trojai-example/output/:rw \
mitigation-test \
--test \
--model_filepath /trojai-example/data/mitigated_model.pt \
--dataset /trojai-example/data/dataset/clean_testset \
--output_dirpath /trojai-example/output \
--num_classes 10