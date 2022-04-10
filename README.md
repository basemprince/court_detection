# Court Detection using Deep Hough Transform Line Priors 

Official implementation for [Deep-Hough-Transform-Line-Priors](https://arxiv.org/abs/2007.09493) (ECCV 2020) 

[Yancong Lin](https://yanconglin.github.io/), and [Silvia Laura Pintea](https://silvialaurapintea.github.io/), and [Jan C. van Gemert](http://jvgemert.github.io/)

E-mail: y.lin-1ATtudelftDOTnl

Vision Lab, Delft University of Technology, the Netherlands

## Introduction

Classical work on line segment detection is knowledge-based; it uses carefully designed geometric priors using either image gradients, pixel groupings, or Hough transform variants. Instead, current deep learning methods do away with all prior knowledge and replace priors by training deep networks on large manually annotated datasets. Here, we reduce the dependency on labeled data by building on the classic knowledge-based priors while using deep networks to learn features. We add line priors through a trainable Hough transform block into a deep network. Hough transform provides the prior knowledge about global line parameterizations, while the convolutional layers can learn the local gradient-like line features. On the Wireframe and York Urban datasets we show that adding prior knowledge improves data efficiency as line priors no longer need to be learned from data.


### Cite Deep Hough-Transform Line Priors

If you find Deep Hough-Transform Line Priors useful in your research, please consider citing:
```bash
@article{lin2020deep,
  title={Deep Hough-Transform Line Priors},
  author={Lin, Yancong and Pintea, Silvia L and van Gemert, Jan C},
  booktitle={EECV 2020},
  year={2020}
}
```
