# fastai에 오신 것을 환영합니다. (fast.ai 한국어 번역 튜토리얼)
> fastai는 최신 모범 사례를 사용하여 빠르고 정확하게 신경망 훈련하는 것을 단순화합니다.


![CI](https://github.com/fastai/fastai/workflows/CI/badge.svg) [![PyPI](https://img.shields.io/pypi/v/fastai?color=blue&label=pypi%20version)](https://pypi.org/project/fastai/#description) [![Conda (channel only)](https://img.shields.io/conda/vn/fastai/fastai?color=seagreen&label=conda%20version)](https://anaconda.org/fastai/fastai) [![Build fastai images](https://github.com/fastai/docker-containers/workflows/Build%20fastai%20images/badge.svg)](https://github.com/fastai/docker-containers) ![docs](https://github.com/fastai/fastai/workflows/docs/badge.svg)

## 설치하기

[Google Colab](https://colab.research.google.com/) 을 사용하면 별도의 설치 없이 fastai를 사용할 수 있습니다. 사실상, 이 문서의 모든 페이지는 대화형 노트북으로도 사용할 수 있습니다.
페이지 상단의 "colab에서 열기"를 클릭하여 열 수 있습니다. (Colab 런타임을 "GPU"로 변경하여 빠르게 실행해야 합니다!)
자세한 내용은 [Colab 사용하기](https://course.fast.ai/start_colab)에 대한 fast.ai 문서를 참조하세요.

Linux 또는 Windows(NB: Mac은 지원되지 않음)를 실행하는 한 conda를 사용하여 사용자의 컴퓨터에 fastai를 설치할 수 있습니다.(권장)
Windows의 경우 중요한 참고 사항은 "Windows에서 실행"을 참조하세요.

만약 [miniconda](https://docs.conda.io/en/latest/miniconda.html) (권장)을 사용한다면 아래의 명령어를 실행하세요. (만약 `conda` 를 사용하는 대신에 [mamba](https://github.com/mamba-org/mamba)를 사용한다면 설치과정이 훨씬 더 빠르고 신뢰할 수 있습니다.):
```bash
conda install -c fastchan fastai
```

...또는 [Anaconda](https://www.anaconda.com/products/individual)를 사용한다면
아래의 명령어를 실행하세요.:
```bash
conda install -c fastchan fastai anaconda
```

To install with pip, use: `pip install fastai`. If you install with pip, you should install PyTorch first by following the PyTorch [installation instructions](https://pytorch.org/get-started/locally/).

If you plan to develop fastai yourself, or want to be on the cutting edge, you can use an editable install (if you do this, you should also use an editable install of [fastcore](https://github.com/fastai/fastcore) to go with it.) First install PyTorch, and then:

``` 
git clone https://github.com/fastai/fastai
pip install -e "fastai[dev]"
``` 

## Learning fastai

The best way to get started with fastai (and deep learning) is to read [the book](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527), and complete [the free course](https://course.fast.ai).

To see what's possible with fastai, take a look at the [Quick Start](https://docs.fast.ai/quick_start.html), which shows how to use around 5 lines of code to build an image classifier, an image segmentation model, a text sentiment model, a recommendation system, and a tabular model. For each of the applications, the code is much the same.

Read through the [Tutorials](https://docs.fast.ai/tutorial) to learn how to train your own models on your own datasets. Use the navigation sidebar to look through the fastai documentation. Every class, function, and method is documented here.

To learn about the design and motivation of the library, read the [peer reviewed paper](https://www.mdpi.com/2078-2489/11/2/108/htm).

## About fastai

fastai is a deep learning library which provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains, and provides researchers with low-level components that can be mixed and matched to build new approaches. It aims to do both things without substantial compromises in ease of use, flexibility, or performance. This is possible thanks to a carefully layered architecture, which expresses common underlying patterns of many deep learning and data processing techniques in terms of decoupled abstractions. These abstractions can be expressed concisely and clearly by leveraging the dynamism of the underlying Python language and the flexibility of the PyTorch library. fastai includes:

- A new type dispatch system for Python along with a semantic type hierarchy for tensors
- A GPU-optimized computer vision library which can be extended in pure Python
- An optimizer which refactors out the common functionality of modern optimizers into two basic pieces, allowing optimization algorithms to be implemented in 4–5 lines of code
- A novel 2-way callback system that can access any part of the data, model, or optimizer and change it at any point during training
- A new data block API
- And much more...

fastai is organized around two main design goals: to be approachable and rapidly productive, while also being deeply hackable and configurable. It is built on top of a hierarchy of lower-level APIs which provide composable building blocks. This way, a user wanting to rewrite part of the high-level API or add particular behavior to suit their needs does not have to learn how to use the lowest level.

<img alt="Layered API" src="nbs/images/layered.png" width="345" style="max-width: 345px">

## Migrating from other libraries

It's very easy to migrate from plain PyTorch, Ignite, or any other PyTorch-based library, or even to use fastai in conjunction with other libraries. Generally, you'll be able to use all your existing data processing code, but will be able to reduce the amount of code you require for training, and more easily take advantage of modern best practices. Here are migration guides from some popular libraries to help you on your way:

- [Plain PyTorch](https://docs.fast.ai/migrating_pytorch)
- [Ignite](https://docs.fast.ai/migrating_ignite)
- [Lightning](https://docs.fast.ai/migrating_lightning)
- [Catalyst](https://docs.fast.ai/migrating_catalyst)

## Windows Support

When installing with `mamba` or `conda` replace `-c fastchan` in the installation with `-c pytorch -c nvidia -c fastai`, since fastchan is not currently supported on Windows.

Due to python multiprocessing issues on Jupyter and Windows, `num_workers` of `Dataloader` is reset to 0 automatically to avoid Jupyter hanging. This makes tasks such as computer vision in Jupyter on Windows many times slower than on Linux. This limitation doesn't exist if you use fastai from a script.

See [this example](https://github.com/fastai/fastai/blob/master/nbs/examples/dataloader_spawn.py) to fully leverage the fastai API on Windows.

## Tests

To run the tests in parallel, launch:

`nbdev_test_nbs` or `make test`

For all the tests to pass, you'll need to install the following optional dependencies:

```
pip install "sentencepiece<0.1.90" wandb tensorboard albumentations pydicom opencv-python scikit-image pyarrow kornia \
    catalyst captum neptune-cli
```

Tests are written using `nbdev`, for example see the documentation for `test_eq`.

## Contributing

After you clone this repository, please run `nbdev_install_git_hooks` in your terminal. This sets up git hooks, which clean up the notebooks to remove the extraneous stuff stored in the notebooks (e.g. which cells you ran) which causes unnecessary merge conflicts.

Before submitting a PR, check that the local library and notebooks match. The script `nbdev_diff_nbs` can let you know if there is a difference between the local library and the notebooks.

- If you made a change to the notebooks in one of the exported cells, you can export it to the library with `nbdev_build_lib` or `make fastai`.
- If you made a change to the library, you can export it back to the notebooks with `nbdev_update_lib`.

## Docker Containers

For those interested in official docker containers for this project, they can be found [here](https://github.com/fastai/docker-containers#fastai).
