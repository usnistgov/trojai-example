docker run --gpus 0 -it \
-v /home/carnejk1/trojai-example-llm-mitigation/data:/trojai-example/data/:ro \
-v /home/carnejk1/trojai-example-llm-mitigation/out:/trojai-example/output/:rw \
mitigation-llm-test \
--model meta-llama/Llama-2-70b-hf \
--dataset /trojai-example/data/reddit_eli5.json \
--output_dirpath /trojai-example/output \
