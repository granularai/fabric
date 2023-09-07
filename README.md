# Code for the paper "Detecting Urban Changes with Recurrent Neural Networks from Multitemporal Sentinel-2 Data" (Accepted in IGARSS 2019)

This is the github repository containing the code for the paper ["Detecting Urban Changes with Recurrent Neural Networks from Multitemporal Sentinel-2 Data"](https://sagarverma.github.io/others/CD_IGARSS_2019.pdf) by Maria Papadomanolaki, Sagar Verma, Maria Vakalopoulou, Siddharth Gupta, Konstantinos Karantzalos.

MultiDate-LSTM code is available [here](https://github.com/mpapadomanolaki/UNetLSTM.git).

## Requirements
The code has been tested on:

- 4xNvidia P100 GPU
- Ubuntu 18.04 LTS on 96 vCPUs and 240 GB of RAM (Large scale inference will be slower on other configurations)
- [Pytorch](https://pytorch.org/) v0.4.0
- Opencv 3.0


## Urbanization detection output
<img src="https://sagarverma.github.io/others/CD_IGARSS_2019.png">

## Multidate Sentinel-2 dataset

1. [OSCD](https://rcdaudt.github.io/oscd/)
2. [OSCD + Our Dates](https://engine.granular.ai/organizations/granular/projects/61916375b40d35cba31b0b29/overview)
3. [Pretrained weight for Bi-Date Siamese Model](https://engine.granular.ai/organizations/granular/projects/61916375b40d35cba31b0b29/overview)


## Contact
For any queries, please contact
```
Sagar Verma: sagar@granular.ai
```
