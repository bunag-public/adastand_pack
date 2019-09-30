# はじめに
こちらはNTTで開発された深層学習の学習を高速化する技術 **Adastand/SDProp** のサンプルコードです。

**Adastand/SDProp** についての詳細は下記を参照ください。

* [NTT技術ジャーナル - 2018 Vol.30 No.6 - 特集 新たなサービス創造 に向けて進化するNTTのAI - 深層学習のための先進的な学習技術] (https://www.ntt.co.jp/journal/1806/files/JN20180630.pdf)
* [NTT Technical Review - August 2018 Vol.16 No.8 - Feature Articles: NTT’s Artificial Intelligence Evolves to Create Novel Services - Advanced Learning Technologies for Deep Learning] (https://www.ntt-review.jp/archive/ntttechnical.php?contents=ntr201808fa6.html)
* [IJCAI-17 - Adaptive Learning Rate via Covariance Matrix Based Preconditioning for Deep Neural Networks] (https://www.ijcai.org/proceedings/2017/267)


# インストール
リポジトリをクローンしてリポジトリのトップディレクトリ(`setup.py` のあるディレクトリ)で下記を実行してください。

```bash
$ pip3 install .
```


# 使い方
各フレームワーク向けのパッケージを下記のようにインポートして使ってください(Adastandの場合)。

## chainer
```python
from dloptimizer.chainer import Adam
optimizer = Adam(alpha=0.01, adastand=True)
```

## pytorch
```python
from dloptimizer.pytorch import Adastand
optimizer = Adastand(lr=0.01)
```

## tensorflow
```python
from dloptimizer.tensorflow import AdastandOptimizer
optimizer = AdastandOptimizer(learning_rate=0.01)
```

## keras in tensorflow
```python
from dloptimizer.tensorflow.keras import Adastand
optimizer = Adastand(lr=0.01)
```

# Citation of SDProp
```
@inproceedings{ida17,
  author    = {Yasutoshi Ida and
               Yasuhiro Fujiwara and
               Sotetsu Iwamura},
  title     = {Adaptive Learning Rate via Covariance Matrix Based Preconditioning for Deep Neural Networks},
  booktitle = {Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence ({IJCAI})},
  pages     = {1923--1929},
  year      = {2017}
}
```
