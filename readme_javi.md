##Commands for train the model

the dataset is in data/config.py copilot_dataset = ..... aqui tienes la referencia para las imagenes y etiquetas

Lo del max_size tambien está en data/config.py en yolact_base_config tmb se puede cambiar de False a True el preserve_aspect_ratio para que no sea max_size x maz_size

enforce_size es donde se cambiaría el aspect ratio para que no sea cuadrada si pones a True ese preserve. 

$ python train.py --config=yolact_base_config --batch_size=2 --dataset=copilot_dataset --validation_epoch=50 --num_workers=4

Para evaluar yolact++, simplemente cambia la configuracion

$ python train.py --config=yolact_plus_base_config --batch_size=2 --dataset=copilot_dataset --validation_epoch=15

##Commands for evaluate the model

$ python eval.py --trained_model=weights/yolact_base_416_10000.pth --score_threshold=0.15 --top_k=15 --images=./data/copilot/images/test2023/:./data/copilot/images/ --output_coco_json --output_web_json/

$ python eval.py --trained_model=weights/yolact_base_416_10000.pth --score_threshold=0.15 --top_k=15 --images=./data/copilot/images/test2023/:./data/copilot/images/test2023/

$ python eval.py --trained_model=results_final/yolact_base_312_10000.pth --score_threshold=0.05 --top_k=500 --images=./data/copilot/images/test2023/:./data/copilot/images/

$ python eval.py --trained_model=weigths/yolact_plus_base_312_10000.pth --score_threshold=0.05 --top_k=500 --images=./data/copilot/images/test2023/:./data/copilot/images/ --config=yolact_plus_base_config
~                                                                                          
