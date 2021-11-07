## Multi-label classification

이 작업에서는 다양한 종류의 오브젝트 또는 사람이 포함된 파스칼 데이터셋(Pascal Dataset)을 사용할 것입니다. 이 데이터셋은 한 가지 종류의 이미지 인스턴스뿐만 아니라 주변의 바운딩 박스까지 탐지할 수 있습니다. 여기에선 주어진 이미지에서 모든 클래스를 예측하고자 합니다.


Multi-label classification은 한 가지 분류에만 속하지 않는다는 점에서 single-label classification의 연장선 상에 있습니다. 한 이미지 안에 사람과 말이 함께 있을 수도 있고, 어떠한 범주에도 속하지 않을 수도 있습니다.


이전과 마찬가지로 데이터셋은 매우 쉽게 다운로드 할 수 있습니다:

```python
path = untar_data(URLs.PASCAL_2007)
path.ls()
```
```bash
(#9) [Path('/home/jhoward/.fastai/data/pascal_2007/valid.json'),Path('/home/jhoward/.fastai/data/pascal_2007/test.json'),Path('/home/jhoward/.fastai/data/pascal_2007/test'),Path('/home/jhoward/.fastai/data/pascal_2007/train.json'),Path('/home/jhoward/.fastai/data/pascal_2007/test.csv'),Path('/home/jhoward/.fastai/data/pascal_2007/models'),Path('/home/jhoward/.fastai/data/pascal_2007/segmentation'),Path('/home/jhoward/.fastai/data/pascal_2007/train.csv'),Path('/home/jhoward/.fastai/data/pascal_2007/train')]
```

각각의 이미지 레이블에 대한 정보는 'train.csv'라는 파일에 담겨 있습니다. pandas를 이용해 불러와 보도록 합시다:
'''python
df = pd.read_csv(path/'train.csv')
df.head()
'''
