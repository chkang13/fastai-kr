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
 
 그런 다음 간단하게 레이블 함수를 정의할 수 있습니다.
 ```
 def label_func(f): return f[0].isupper()
 ```
 
모델에 대한 데이터를 준비하려면 [DataLoaders](https://docs.fast.ai/data.core.html#DataLoaders) 객체에 데이터를 넣어야 합니다. 파일 이름을 사용하여 레이블을 지정하는 함수인 [ImageDataLoaders.from_name_func](https://docs.fast.ai/vision.data.html#ImageDataLoaders.from_name_func)를 사용합니다. 문제에 더 적합할 수 있는 ImageDataLoaders의 다른 공장 방법이 있으므로 vision.data에 있는 모든 것들을 확인하세요. 
```
dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224)) 
```
작업 중인 디렉터리의 경로, 가져온 파일들, label_func 함수 및 마지막 파라미터로 item_tfms를 이 함수에 전달했습니다. 이것은 데이터셋에 있는 모든 항목들에 대해 이미지를 224 x 224로 크기를 조정하는 Transform입니다. 사각형으로 만들기 위해 가장 큰 치원에 대해 random crop을 적용하고 그 다음 224 x 224로 크기를 조정합니다. 만약 이것을 전달하지 못했다면 항목들을 일괄적으로 묶는 것은 불가능하기 때문에 나중에 오류를 얻게 될 것입니다.
 
그런 다음 show_batch 메서드에서 모든 것이 정상적으로 보이는지 확인할 수 있습니다(참은 고양이용, 거짓은 개용).
```
dls.show_batch()
```

이미지

그런 다음 Learner를 만들 수 있습니다. Learner는 데이터와 훈련용 모델을 결합하고 전송 학습을 사용하여 두 줄의 코드로 사전 훈련된 모델을 미세하게 조정하는 fastai 객체입니다.
```
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```
표

표



