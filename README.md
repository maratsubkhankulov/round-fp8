# FP8 stochastic rounding

An implementation bfloat16 to fp8 quantization using stochastic rounding based on this [article](https://nhigham.com/2020/07/07/what-is-stochastic-rounding/)

## Tests
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m pytest 
```
