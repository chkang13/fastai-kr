# 다중 레이블 분류

이 작업에서는 다양한 종류의 오브젝트 또는 사람이 포함된 파스칼 데이터셋([Pascal Dataset](http://host.robots.ox.ac.uk/pascal/VOC/))을 사용할 것입니다. 이 데이터셋은 한 가지 종류의 이미지 인스턴스뿐만 아니라 주변의 바운딩 박스까지 탐지할 수 있습니다. 여기에선 주어진 이미지에서 모든 클래스를 예측하고자 합니다.


다중 레이블 분류는 한 가지 분류에만 속하지 않는다는 점에서 single-label classification의 연장선 상에 있습니다. 한 이미지 안에 사람과 말이 함께 있을 수도 있고, 어떠한 범주에도 속하지 않을 수도 있습니다.


이전과 마찬가지로 데이터셋은 매우 쉽게 다운로드 할 수 있습니다:

```python
path = untar_data(URLs.PASCAL_2007)
path.ls()
```
```
(#9) [Path('/home/jhoward/.fastai/data/pascal_2007/valid.json'),Path('/home/jhoward/.fastai/data/pascal_2007/test.json'),Path('/home/jhoward/.fastai/data/pascal_2007/test'),Path('/home/jhoward/.fastai/data/pascal_2007/train.json'),Path('/home/jhoward/.fastai/data/pascal_2007/test.csv'),Path('/home/jhoward/.fastai/data/pascal_2007/models'),Path('/home/jhoward/.fastai/data/pascal_2007/segmentation'),Path('/home/jhoward/.fastai/data/pascal_2007/train.csv'),Path('/home/jhoward/.fastai/data/pascal_2007/train')]
```

각각의 이미지 레이블에 대한 정보는 `train.csv`라는 파일에 담겨 있습니다. pandas를 이용해 불러와 보도록 합시다:
```python
df = pd.read_csv(path/'train.csv')
df.head()
```
||fname|labels|is_valid|
|:---:|:---|:---|:---|
|**0**|000005.jpg|chair|True
|**1**|000007.jpg|car|True
|**2**|000009.jpg|horse person|True
|**3**|000012.jpg|car|False
|**4**|000016.jpg|bicycle|True

## 다중 레이블 분류 - High-level API의 사용
이는 꽤나 직관적입니다: 각각의 파일에 대해 서로 다른 레이블들(스페이스로 분류된)을 얻고 마지막 열은 그 데이터의 검증 집합(validation set) 포함 여부를 알려줍니다. [DataLoaders](https://docs.fast.ai/data.core.html#DataLoaders)에서 이들을 빨리 얻기 위해서 `from_df`라는 팩토리 메소드를 사용합니다. 모든 이미지가 있는 기본 경로, 기본 경로와 파일 이름 사이에 추가할 폴더 (여기선 `train`), 검증 집합을 위한 `valid_col` (지정하지 않은 경우 임의의 하위 집합을 선택), 레이블을 분류하기 위한 `label_delim`, 그리고 이전처럼 `item_tfms`와 `batch_tfms`을 지정할 수 있습니다.

`fn_col`과 `label_col`은 기본적으로 각각 첫 번째 열과 두 번째 열로 설정되므로 따로 지정해줄 필요가 없습니다.

```python
dls = ImageDataLoaders.from_df(df, path, folder='train', valid_col='is_valid', label_delim=' ',
                               item_tfms=Resize(460), batch_tfms=aug_transforms(size=224))
```
이전처럼 `show_batch` 메소드를 통해 데이터를 확인할 수 있습니다.
```python
dls.show_batch()
```
![image1](https://user-images.githubusercontent.com/68338919/140639987-b6c72f1a-75d0-47f9-9aa5-9e0039b4e264.png)

모델을 학습은 이전처럼 하면 쉽습니다: 동일한 함수를 사용할 수 있고 fastai 라이브러리는 자동적으로 multi-label problem을 탐지해줍니다. 따라서 올바른 손실 함수가 선택됩니다. 유일한 차이점은 사용되는 메트릭스입니다:

[error_rate](https://docs.fast.ai/metrics.html#error_rate)는 multi-label problem을 해결해주지 않지만 대신 `accuracy_thresh`를 사용할 수 있습니다.
```python
learn = cnn_learner(dls, resnet50, metrics=partial(accuracy_multi, thresh=0.5))
```
전과 마찬가지로 `learn.lr_find`는 좋은 학습률을 보여줍니다:
```python
learn.lr_find()
```
```
SuggestedLRs(lr_min=0.025118863582611083, lr_steep=0.03981071710586548)
```
![image2](https://user-images.githubusercontent.com/68338919/140639991-d1154ca3-5f84-4851-a8a8-789e9c964ca7.png)

다양한 학습률과 잘 조정이 된 학습 모델을 얻을 수 있습니다:
```python
learn.fine_tune(2, 3e-2)
```
|epoch|train_loss|valid_loss|accuracy_multi|time|
|:---|:---|:---|:---|:--|
|0|0.437855|0.136942|0.954801|00:17

|epoch|train_loss|valid_loss|accuracy_multi|time|
|:---|:---|:---|:---|:--|
0|0.156202|0.465557|0.914801|00:20|
1|0.179814|0.382907|0.930040|00:20|
2|0.157007|0.129412|0.953924|00:20|
3|0.125787|0.109033|0.960856|00:19|

결과는 다음과 같습니다:
```python
learn.show_results()
```
![image3](https://user-images.githubusercontent.com/68338919/140639994-20444f8c-ab86-4f93-bf1a-617ccd1a0f26.png)

또는 주어진 이미지를 통해 추정할 수 있습니다:
```python
learn.predict(path/'train/000005.jpg')
```
```
((#2) ['chair','diningtable'],
 TensorImage([False, False, False, False, False, False, False, False,  True, False,
          True, False, False, False, False, False, False, False, False, False]),
 TensorImage([1.6750e-03, 5.3663e-03, 1.6378e-03, 2.2269e-03, 5.8645e-02, 6.3422e-03,
         5.6991e-03, 1.3682e-02, 8.6864e-01, 9.7093e-04, 6.4747e-01, 4.1217e-03,
         1.2410e-03, 2.9412e-03, 4.7769e-01, 9.9664e-02, 4.5190e-04, 6.3532e-02,
         6.4487e-03, 1.6339e-01]))
```
단일 분류 예측(single classification predictions)에서는 세 가지를 알아냈었습니다. 마지막은 각 클래스에 대한(0에서 1까지) 모델의 추정입니다. 두 번째에서 마지막까지가 원핫(one-hot) 인코딩에 해당하고(확률이 0.5를 초과하면 모든 추정에서 `True`) 첫번째가 읽을 수 있는 버전으로 디코딩됩니다.

이제 어디에서 문제가 발생했는지 확인할 수 있습니다:
```python
interp = Interpretation.from_learner(learn)
interp.plot_top_losses(9)
```
||target|predicted|probabilities|loss|
|:---|:---|:---|:---|:--|
**0**|car;person;tvmonitor|car|tensor([7.2388e-12, 5.9609e-06, 1.7054e-11, 3.8985e-09, 7.7078e-12, 3.4044e-07,\n 9.9999e-01, 7.2118e-12, 1.0105e-05, 3.1035e-09, 2.3334e-09, 9.1077e-09,\n 1.6201e-09, 1.1083e-08, 1.0809e-02, 2.1072e-07, 9.5961e-16, 5.0478e-07,\n 4.4531e-10, 9.6444e-12])|1.494603157043457|
**1**|boat|car|tensor([8.3430e-06, 1.9416e-03, 6.9865e-06, 1.2985e-04, 1.6142e-06, 8.2200e-05,\n 9.9698e-01, 1.3143e-06, 1.0047e-03, 4.9794e-05, 1.9155e-05, 4.7409e-05,\n 7.5056e-05, 1.6572e-05, 3.4760e-02, 6.9266e-04, 1.3006e-07, 6.0702e-04,\n 1.5781e-05, 1.9860e-06])|0.7395917773246765
**2**|bus;car|car|tensor([2.2509e-11, 1.0772e-05, 6.0177e-11, 4.8728e-09, 1.7920e-11, 4.8695e-07,\n 9.9999e-01, 9.0638e-12, 1.9819e-05, 8.8023e-09, 5.1272e-09, 2.3535e-08,\n 6.0401e-09, 7.2609e-09, 4.4117e-03, 4.8268e-07, 1.2528e-14, 1.2667e-06,\n 8.2282e-10, 1.6300e-11])|0.7269787192344666|
**3**|chair;diningtable;person|person;train|tensor([1.6638e-03, 2.0881e-02, 4.7525e-03, 2.6422e-02, 6.2972e-04, 4.7170e-02,\n 1.2263e-01, 2.9744e-03, 5.5352e-03, 7.1830e-03, 1.0062e-03, 2.6123e-03,\n 1.8208e-02, 5.9618e-02, 7.6859e-01, 3.3504e-03, 1.1324e-03, 2.3881e-03,\n 6.5440e-01, 1.7040e-03])|0.6879587769508362|
**4**|boat;chair;diningtable;person|person|tensor([0.0058, 0.0461, 0.0068, 0.1083, 0.0094, 0.0212, 0.4400, 0.0047, 0.0166,\n 0.0054, 0.0030, 0.0258, 0.0020, 0.0800, 0.5880, 0.0147, 0.0026, 0.1440,\n 0.0219, 0.0166])|0.6826764941215515|
**5**|bicycle;car;person|car|tensor([3.6825e-09, 7.3755e-05, 1.7181e-08, 4.5056e-07, 3.5667e-09, 1.0882e-05,\n 9.9939e-01, 6.0704e-09, 5.7179e-05, 3.8519e-07, 9.3825e-08, 6.1463e-07,\n 3.9191e-07, 2.6800e-06, 3.3091e-02, 3.1972e-06, 2.6873e-11, 1.1967e-05,\n 1.1480e-07, 3.3320e-09])|0.6461981534957886|
**6**|bottle;cow;person|chair;person;sofa|tensor([5.4520e-04, 4.2805e-03, 2.3828e-03, 1.4127e-03, 4.5856e-02, 3.5540e-03,\n 9.1525e-03, 2.9113e-02, 6.9326e-01, 1.0407e-03, 7.0658e-02, 3.1101e-02,\n 2.4843e-03, 2.9908e-03, 8.8695e-01, 2.2719e-01, 1.0283e-03, 6.0414e-01,\n 1.3598e-03, 5.7382e-02])|0.6329519152641296|
**7**|chair;dog;person|cat|tensor([3.4073e-05, 1.3574e-03, 7.0516e-04, 1.9189e-04, 6.0819e-03, 4.7242e-05,\n 9.6424e-04, 9.3669e-01, 9.0736e-02, 8.1472e-04, 1.1019e-02, 5.4633e-02,\n 2.6190e-04, 1.4943e-04, 1.2755e-02, 1.7530e-02, 2.2532e-03, 2.2129e-02,\n 1.5532e-04, 6.6390e-03])|0.6249645352363586|
**8**|car;person;pottedplant|car|tensor([1.3978e-06, 2.1693e-03, 2.2698e-07, 7.5037e-05, 9.4007e-07, 1.2369e-03,\n 9.9919e-01, 1.0879e-07, 3.1837e-04, 1.8340e-05, 7.5422e-06, 2.3891e-05,\n 2.5957e-05, 3.0890e-05, 8.4529e-02, 2.0280e-04, 4.1234e-09, 1.7978e-04,\n 2.3258e-05, 6.0897e-07])|0.5489450693130493|

![image4](https://user-images.githubusercontent.com/68338919/140639999-8a68cddd-326a-4066-bea5-965eaad4c532.png)

## 다중 레이블 분류 - 데이터 블럭 API를 통해
[DataLoaders](https://docs.fast.ai/data.core.html#DataLoaders)에서 데이터를 얻기 위해 데이터 블럭 API를 사용할 수 있습니다. 전에서 언급했다시피 아직 새로운 API에 대해 배우기가 힘들다면 이 파트는 넘어가도 괜찮습니다.

데이터프레임에서 데이터가 어떻게 구조화되어 있는지 생각해봅시다:
```python
df.head()
```
||fname|labels|is_valid|
|:---:|:---|:---|:---|
|**0**|000005.jpg|chair|True
|**1**|000007.jpg|car|True
|**2**|000009.jpg|horse person|True
|**3**|000012.jpg|car|False
|**4**|000016.jpg|bicycle|True

이 경우에는 다음을 통해 데이터 블록을 구축합니다:

- 사용되는 방법: [ImageBlock](https://docs.fast.ai/vision.data.html#ImageBlock)과 [MultiCategoryBlock](https://docs.fast.ai/data.block.html#MultiCategoryBlock).
- 데이터 프레임에서 입력 항목을 가져오는 방법: 여기서는 `fname` 열을 읽고 적절한 파일 이름을 얻으려면 처음에 path/train/을 추가해야 합니다.
- 데이터 프레임에서 대상을 가져오는 방법: 여기서는 `labels`을 읽고 공백을 기준으로 분할해야 합니다.
- `is_valid`를 통해 항목들을 나눌 수 있습니다.
- `item_tfms`와 `batch_tfms`는 이전과 같습니다.
```python
pascal = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=ColSplitter('is_valid'),
                   get_x=ColReader('fname', pref=str(path/'train') + os.path.sep),
                   get_y=ColReader('labels', label_delim=' '),
                   item_tfms = Resize(460),
                   batch_tfms=aug_transforms(size=224))
```
이 부분은 지금까지와는 조금 다릅니다: 우리가 제공할 데이터 프레임은 이미 모든 항목을 가지고 있기 때문에 모든 항목을 수집하는 기능을 사용할 필요가 없습니다. 그러나 인풋을 가져오려면 해당 데이터 프레임의 행을 전처리해야 하므로 `get_x`를 사용해야 합니다. fastai function 에서 기본으로 제공되는 `noop`을 사용하면 됩니다.

`pascal`은 초안일 뿐입니다. [DataLoaders](https://docs.fast.ai/data.core.html#DataLoaders)를 사용하려면 데이터 소스를 전송해야 합니다:
```python
dls = pascal.dataloaders(df)
```
그러면 `dls.show_batch()`를 통해 몇몇 사진들을 확인할 수 있습니다.
```python
dls.show_batch(max_n=9)
```
![image5](https://user-images.githubusercontent.com/68338919/140640006-8ad12a8c-71bf-420f-863f-2b63e1b1577d.png)
