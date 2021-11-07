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
 
모델에 대한 데이터를 준비하려면 [DataLoaders](https://docs.fast.ai/data.core.html#DataLoaders) 객체에 데이터를 넣어야 합니다. 파일 이름을 사용하여 레이블을 지정하는 함수인 [ImageDataLoaders.from_name_func](https://docs.fast.ai/vision.data.html#ImageDataLoaders.from_name_func)를 사용합니다. 당신의 문제에 더 적합할 수 있는 ImageDataLoaders의 다른 factory method가 있으므로 vision.data에 있는 모든 것들을 확인하세요. 
```
dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224)) 
```
작업 중인 디렉터리의 경로, 가져온 파일들, label_func 함수 및 마지막 파라미터로 item_tfms를 이 함수에 전달했습니다. 이것은 데이터셋에 있는 모든 항목들에 대해 이미지를 224 x 224 크기로 조정하는 Transform입니다. 사각형으로 만들기 위해 가장 큰 치원에 대해 random crop을 적용하고 그 다음 224 x 224로 크기를 조정합니다. 만약 이것을 전달하지 못했다면 항목들을 일괄적으로 묶는 것은 불가능하기 때문에 나중에 오류를 얻게 됩니다.
 
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

첫 번째 라인은 ImageNet에서 사전 훈련을 받은 ResNet34라는 모델을 다운로드하여 특정 문제에 맞게 수정했습니다. 그런 다음 해당 모델을 미세 조정했고, 비교적 짧은 시간 내에 오류율이 0.3%인 모델을 얻게 됩니다. 놀랍습니다!

새로운 이미지를 예측하려면, learn.predict를 사용합니다.:
```
learn.predict(files[0])
```
```
('False', TensorImage(0), TensorImage([9.9998e-01, 2.0999e-05]))
```

예측 방법은 디코딩된 예측(여기서 개의 경우 거짓), 예측 클래스의 색인 및 인덱스 레이블의 모든 클래스의 확률 텐서(이 경우 모델은 개의 값에 대해 상당히 일관성이 있음)의 세 가지를 반환합니다. 이 방법은 파일 이름, PIL 이미지 또는 텐서를 이 경우 직접 받아들인다. show_results 방법을 사용하여 몇 가지 예측을 살펴볼 수도 있습니다.
```
learn.show_results()
```

이미지

텍스트, 표 형식 또는 이 튜토리얼에서 다루는 다른 응용 프로그램을 확인해 보십시오. 그러면 모두 일관된 API를 공유하여 데이터를 수집하고 학습자를 만들고 모델을 교육하고 몇 가지 예측을 살펴봅니다.

# 품종 분류하기

데이터에 품종 이름을 붙이기 위해 정규식을 사용하여 파일 이름에서 추출할 것입니다. 파일 이름을 다시 살펴보면 다음과 같습니다.
```
files[0].name
```
```
'great_pyrenees_173.jpg'
```
마지막 앞에는 클래스가 전부이고, 그 뒤에는 몇 자리 숫자가 나옵니다. 이름을 나타내는 정규식은 다음과 같습니다.
```
pat = r'^(.*)_\d+.jpg'
```
정규식을 사용하여 데이터에 레이블을 지정하는 것이 매우 일반적이므로(일반적으로 파일 이름에 레이블이 숨겨져 있음) 다음과 같은 작업을 수행하는 factory ethod가 있습니다.

```
dls = ImageDataLoaders.from_name_re(path, files, pat, item_tfms=Resize(224))
```

이전과 마찬가지로 show_batch를 사용하여 데이터를 살펴볼 수 있습니다.

```
dls.show_batch()
```

이미지

37종의 다른 품종 간에 고양이 또는 개의 정확한 품종을 분류하는 것이 더 어려운 문제이기 때문에 데이터 증가를 사용하기 위해 DataLoader의 정의를 약간 변경할 것입니다.

```
dls = ImageDataLoaders.from_name_re(path, files, pat, item_tfms=Resize(460),
                                    batch_tfms=aug_transforms(size=224))
```

이번에는 배치하기 전에 더 큰 크기로 크기를 조정했으며 batch_tfms를 추가했습니다. ug_transforms는 많은 데이터 세트에서 잘 작동하는 기본값이 포함된 데이터 증강 변환 컬렉션을 제공하는 함수입니다. 적절한 인수를 oug_transforms에 전달하여 이러한 변환을 사용자 정의할 수 있습니다.
```
dls.show_batch()
```

이미지

그런 다음 이전과 동일하게 Learner를 만들고 모델을 훈련시킬 수 있습니다.
```
learn = cnn_learner(dls, resnet34, metrics=error_rate)
```

우리는 이전에 기본 학습률을 사용했지만, 가능한 최고의 학습률을 찾고 싶을 수도 있습니다. 이를 위해 학습 속도 측정기를 사용할 수 있습니다.
```
learn.lr_find()
```
```
SuggestedLRs(lr_min=0.010000000149011612, lr_steep=0.0063095735386013985)
```

그래프

이 그래프는 학습 속도 측정기의 그래프를 표시하고 두 가지 제안(최소를 10으로 나눈 것과 가장 가파른 기울기)을 제공합니다. 여기서는 3e-3을 사용합시다. 또한 epoch를 좀 더 진행할 것입니다.

```
learn.fine_tune(2, 3e-3)
```

표


표

show_results를 사용하여 몇 가지 예측을 살펴볼 수 있습니다.
```
learn.show_results()
```

이미지

또 다른 유용한 것은 해석 객체로서, 모델이 어디에서 더 나쁜 예측을 했는지 보여줄 수 있습니다.
```
interp = Interpretation.from_learner(learn)
```
```
interp.plot_top_losses(9, figsize=(15,10))
```

이미지

# 단일 레이블 분류 - data block API를 통해

data block API를 사용하여 DataLoader에서 데이터를 가져올 수도 있습니다. 이것은 좀 더 심화된 것이므로, 만약 아직 새로운 API를 배우는 것이 어렵다면 이 부분을 자유롭게 건너뛸 수 있습니다.

datablock은 fastai 라이브러리에 다음과 같은 많은 정보를 제공하여 구축됩니다.

- block이라는 인자를 통해 사용되는 유형: 이미지와 카테고리가 있으므로 ImageBlock과 CategoryBlock을 전달합니다.
- 원시 데이터를 가져오는 방법, 함수 get_image_files.
- 항목에 레이블을 지정하는 방법(여기서는 이전과 동일한 정규식)을 사용합니다.
- 항목들을 분할하는 방법, random splitter.
- item_tfms 및 batch_tfms, 이전과 유사.

```
pets = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                 get_items=get_image_files, 
                 splitter=RandomSplitter(),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224))
```

pets 객체 자체는 비어 있습니다. 데이터 수집에 도움이 되는 기능만 포함되어 있습니다. DataLoaders를 얻으려면 dataloaders 메서드를 호출해야 합니다. 데이터의 출처를 전달합니다.:

```
dls = pets.dataloaders(untar_data(URLs.PETS)/"images")
```
그러면 dls.show_batch()를 통해 사진 몇 장을 볼 수 있습니다.
```
dls.show_batch(max_n=9)
```

이미지











