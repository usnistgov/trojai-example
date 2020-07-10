#python fake_trojan_detector.py --model_filepath=./model.pt --result_filepath=./output.txt --scratch_dirpath=./scratch/ --examples_dirpath=./example/

DATA_FOLDER=$HOME/data/trojai-round0-dataset
MODEL_FILEPATH=$DATA_FOLDER/id-00000000/model.pt
EXAMPLES_DIRPATH=$DATA_FOLDER/id-00000000/example_data/

python trojan_detector.py --model_filepath=$MODEL_FILEPATH --examples_dirpath=$EXAMPLES_DIRPATH
