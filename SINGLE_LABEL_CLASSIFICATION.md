# Computer vision
> Computer vision에 fastai 라이브러리 사용하기

이 튜토리얼은 대부분의 컴퓨터 비전 작업에서 [Learner](https://docs.fast.ai/learner.html#Learner)를 빠르게 구축하고 사전 훈련된 모델을 미세 조정하는 방법을 강조합니다.

# Single-label classification (단일 레이블 분류)
이 작업을 위해 서로 다른 37가지 품종의 고양이와 개의 이미지가 포함된 [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)을 사용합니다. 먼저 간단한 cat-vs-dog 분류기를 구축하는 방법을 보여주고 모든 품종을 분류할 수 있는 좀 더 심화된 모델을 보여줍니다.

다음 코드를 통해 데이터 세트를 다운로드하고 압축을 풀 수 있습니다.
```
path = untar_data(URLs.PETS)
```

이 명령어는 다운로드를 한 번만 수행하고 압축 해제된 아카이브의 위치를 반환합니다. [.ls()] 메서드를 통해 내부에 무엇이 있는지 확인할 수 있습니다.
```
path.ls()
```

지금은 주석 폴더를 무시하고 이미지 폴더에 집중하겠습니다. [get_image_files](https://docs.fast.ai/data.transforms.html#get_image_files)은 하나의 폴더에서 모든 이미지 파일을 (재귀적으로) 가져오는 데 도움을 주는 fastai 함수입니다.
```
files = get_image_files(path/"images")
len(files)
```
```
7390
```

# 고양이 vs 개
고양이 대 개 문제에 대한 데이터에 레이블을 지정하려면 어떤 파일 이름이 개 사진이고 어떤 파일 이름이 고양이 사진인지 알아야 합니다. 구별하는 쉬운 방법이 있습니다: 파일 이름은 고양이의 경우 대문자로 시작하고 개는 소문자로 시작합니다.
```
files[0],files[6]
```
```
(Path('/home/jhoward/.fastai/data/oxford-iiit-pet/images/great_pyrenees_173.jpg'),
 Path('/home/jhoward/.fastai/data/oxford-iiit-pet/images/staffordshire_bull_terrier_173.jpg'))
 ```
 
 그런 다음 쉬운 레이블 함수를 정의할 수 있습니다.
 ```
 def label_func(f): return f[0].isupper()
 ```
 
모델에 대한 데이터를 준비하려면 [DataLoaders](https://docs.fast.ai/data.core.html#DataLoaders) 객체에 데이터를 넣어야 합니다. 파일 이름을 사용하여 레이블을 지정하는 함수인 [ImageDataLoaders.from_name_func](https://docs.fast.ai/vision.data.html#ImageDataLoaders.from_name_func)를 사용합니다. 문제에 더 적합할 수 있는 ImageDataLoaders의 다른 공장 방법이 있으므로 vision.data에 있는 모든 것들을 확인하세요. 
```
dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))
```

 
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
