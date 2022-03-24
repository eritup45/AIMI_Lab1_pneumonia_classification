# AIMI LAB1 Pneumonia Classification

## Introduction


## Data preparation
* Download data: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

* Organize data path:
```
archive/chest_xray
```

## Train and Test
`python sample.py`

## Test only
`python inference.py`

## Pretrained model
[Download Link](https://drive.google.com/drive/folders/10xKR3VQxdE7HuiFqWc0ROIY9nFdRfdiR?usp=sharing)

## Model architechure
resnext50

## Results
* Accuracy: 94.39%
* F1-score: 0.9556

<table>
  <thead>
    <tr>
      <th>Training accuracy</th>
      <th>Testing accuracy</th>
      <th>Testing F1-score</th>
    </tr>
   </thead>
   <tbody>
     <tr>
       <td><img src="https://i.imgur.com/J7jl8wq.png"></td>
       <td> <img src="https://i.imgur.com/zoWHdcD.png"> </td>
       <td><img src="https://i.imgur.com/oBsW0hI.png"></td>
     </tr>

  </tbody>
</table>

* Test Confusion matrix:
![](https://i.imgur.com/wC3DIG3.png)







