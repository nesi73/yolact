
#commands to train the model

#commands to evaluate the model
$ python eval.py --trained_model=weights/yolact_base_416_10000.pth --score_threshold=0.15 --top_k=15 --images=./data/copilot/images/test2023/:./data/copilot/images/ --output_coco_json --output_web_json/