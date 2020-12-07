#python fake_trojan_detector.py --model_filepath=./model.pt --result_filepath=./output.txt --scratch_dirpath=./scratch/ --examples_dirpath=./example/

MD_NAME=id-00001003
DATA_FOLDER=$HOME/data/round2-dataset-train
MODEL_FILEPATH=$DATA_FOLDER/$MD_NAME/model.pt
EXAMPLES_DIRPATH=$DATA_FOLDER/$MD_NAME/poisoned_example_data/

python trojan_detector.py --model_filepath=$MODEL_FILEPATH --examples_dirpath=$EXAMPLES_DIRPATH
