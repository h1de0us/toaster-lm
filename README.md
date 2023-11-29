## TOASTER-LM: Tiny Omniscient and Almost Sentient TransformER-based Language Model

### Prerequisites:
To install all the required dependencies, run
```
pip install -r requirements.txt
```
### How to train a model or generate some stories
To download the data and parse it into one text file, run:
```
chmod +x run.sh
./run.sh
```

To train a model with the default parameters, use
```
python3 train.py
```
P.S: Hydra-based config parser is on the way

### Architecture
To learn more about the used architecture and its particular qualities, read the [report](https://wandb.ai/h1de0us/LLM-Homework/reports/Toaster-LM-Report--Vmlldzo2MTE2Nzc5)
