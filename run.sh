#python fake_trojan_detector.py --model_filepath=./model.pt --result_filepath=./output.txt --scratch_dirpath=./scratch/ --examples_dirpath=./example/

DATA_FOLDER=$HOME/data/round2-dataset-train
MODEL_FILEPATH=$DATA_FOLDER/id-00000001/model.pt
EXAMPLES_DIRPATH=$DATA_FOLDER/id-00000001/example_data/

python trojan_detector.py --model_filepath=$MODEL_FILEPATH --examples_dirpath=$EXAMPLES_DIRPATH
