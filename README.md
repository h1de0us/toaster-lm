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
./run.sh download data
cd .. # move to toaster-lm
./run.sh parse_stories
```
Then just run cells in main.ipynb.


### Architecture
To learn more about the used architecture and its particular qualities, read the [report](https://wandb.ai/h1de0us/LLM-Homework/reports/Toaster-LM-Report--Vmlldzo2MTE2Nzc5)
