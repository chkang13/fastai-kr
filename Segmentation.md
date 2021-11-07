# 분할 추출

분할 추출은 이미지의 각 픽셀을 가리키는 항목을 예측하는 것에 어려움이 있습니다. 이 작업을 위해 자동차를 찍은 이미지를 데이터로 가지고 있는 [Camvid dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) 을 사용할 것입니다. 이미지의 각 픽셀은 "road", "car", "pedestrian 와 같은 레이블을 가지고 있습니다.

[untar_data](https://docs.fast.ai/data.external.html#untar_data) 함수를 이용해서 데이터를 다운로드 할 수 있습니다.

```
path = untar_data(URLs.CAMVID_TINY)
path.ls()
```
```
(#3) [Path('/home/jhoward/.fastai/data/camvid_tiny/codes.txt'),Path('/home/jhoward/.fastai/data/camvid_tiny/images'),Path('/home/jhoward/.fastai/data/camvid_tiny/labels')]
```

images 폴더는 이미지를 포함하고 있고, 그에 따른 분할 추출 마스크의 항목은 labels 폴더 안에 있습니다. codes 파일에는 클래스에 따른 정수 값이 들어 있습니다.
```
codes = np.loadtxt(path/'codes.txt', dtype=str)
codes
```
```
array(['Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car',
       'CartLuggagePram', 'Child', 'Column_Pole', 'Fence', 'LaneMkgsDriv',
       'LaneMkgsNonDriv', 'Misc_Text', 'MotorcycleScooter', 'OtherMoving',
       'ParkingBlock', 'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk',
       'SignSymbol', 'Sky', 'SUVPickupTruck', 'TrafficCone',
       'TrafficLight', 'Train', 'Tree', 'Truck_Bus', 'Tunnel',
       'VegetationMisc', 'Void', 'Wall'], dtype='<U17')
```

# 분할 추출 – 높은 수준의 API를 이용하는 경우
이전처럼 [get_image_files](https://docs.fast.ai/data.transforms.html#get_image_files) 함수는 모든 이미지 파일이름을 불러올 수 있게 합니다:
```
fnames = get_image_files(path/"images")
fnames[0]
```
```
Path('/home/jhoward/.fastai/data/camvid_tiny/images/0006R0_f02910.png')
```
레이블 폴더를 자세히 살펴보면:
```
(path/"labels").ls()[0]
```
```
Path('/home/jhoward/.fastai/data/camvid_tiny/labels/0016E5_08137_P.png')
```
분할 추출 마스크가 이미지들의 기본 이름에 _P를 한 것과 같다고 할 수 있기 때문에, 아래처럼 레이블 함수를 정의할 수 있습니다:
```
def label_func(fn): return path/"labels"/f"{fn.stem}_P{fn.suffix}"
```
그렇게 하면 SegmentationDataLoaders를 사용해서 데이터를 얻을 수 있습니다:
```
dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames = fnames, label_func = label_func, codes = codes
)
```

모든 이미지가 같은 크기이기 때문에 이미지의 크기를 재조정하기 위해 item_tfms을 넘길 필요가 없습니다.

 show_batch 함수를 이용하면 이미지 데이터를 확인할 수 있습니다. 이 사례에서는, fastai library가 픽셀당 하나의 특정 색상으로 마스크를 중첩하고 있습니다:
 ```
dls.show_batch(max_n=6)
```
 
기존의CNN 은 분할 추출 시에 작동하지 않기 때문에, UNet이라는 특별한 모델을 사용해야 해서, Learner을 정의하기 위해unet_learner 을 사용합니다:
```
learn = unet_learner(dls, resnet34)
learn.fine_tune(6)
```
|epoch|train_loss|valid_loss|time|
|-----|---|---|---|
|0	|2.802264	|2.476579	|00:03|
|epoch	|train_loss	|valid_loss	|time|
|0	|1.664625	|1.525224	|00:03|
|1	|1.440311	|1.271917	|00:02|
|2	|1.339473	|1.123384	|00:03|
|3	|1.233049	|0.988725	|00:03|
|4	|1.110815	|0.805028	|00:02|
|5	|1.008600	|0.815411	|00:03|
|6	|0.924937	|0.755052	|00:02|
|7	|0.857789	|0.769288	|00:03|

show_results를 통해 예측된 결과에서 특징을 얻을 수 있습니다.
```
learn.show_results(max_n=6, figsize=(7,8))
 ```
 
[SegmentationInterpretation](https://docs.fast.ai/interpret.html#SegmentationInterpretation) 클래스를 사용하여 모델의 오류를 정렬한 후 검증 손실에 대해 k개의 가장 높은 기여도를 가진 사례를 구성할 수 있습니다.
```
interp = SegmentationInterpretation.from_learner(learn)
interp.plot_top_losses(k=3)
```
 
# 분할 추출 – 데이터 블록 API를 사용할 경우
[DataLoaders](https://docs.fast.ai/data.core.html#DataLoaders) 안에 있는 데이터를 얻기 위해 데이터 블록 API를 사용할 수도 있습니다. 전에 언급했듯이, 아직 새로운 API를 배우는 것이 익숙하지 않다면 이 부분을 건너 뛰어도 됩니다.

이 경우 다음과 같은 것을 제공하여 데이터 블록을 구성합니다:
•	사용된 유형: [ImageBlock](https://docs.fast.ai/vision.data.html#ImageBlock) 과  [MaskBlock](https://docs.fast.ai/vision.data.html#MaskBlock). 데이터레서 추측할 수 있는 방법이 없으므로 [MaskBlock](https://docs.fast.ai/vision.data.html#MaskBlock) 에서 코드를 제공.
•	정보를 모으는 방법: [get_image_files](https://docs.fast.ai/data.transforms.html#get_image_files).
•	정보로부터 목표 데이터를 얻는 방법: label_func.
•	정보 분리는 무작위로 진행.
•	batch_tfms 는 데이터 증가를 위해 사용.
```
camvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items = get_image_files,
                   get_y = label_func,
                   splitter=RandomSplitter(),
                   batch_tfms=aug_transforms(size=(120,160)))
dls = camvid.dataloaders(path/"images", path=path, bs=8)
dls.show_batch(max_n=6)
```

# 포인트
이 섹션은 데이터 블록 API를 사용하므로 이전에 건너뛰었다면 이 섹션도 건너뛰는 것을 권장 드립니다.
이제 그림에서 포인트를 예측하는 작업을 살펴보겠습니다. 이 과정을 위해[Biwi Kinect Head Pose Dataset](Biwi Kinect Head Pose Dataset)을 사용할 것입니다. 먼저 평소와 같이 데이터 셋을 다운로드하여 시작하겠습니다.
```
path = untar_data(URLs.BIWI_HEAD_POSE)
```
어떤 결과가 나왔는지 확인해 보겠습니다!
```
path.ls()
```
```
(#50) [Path('/home/sgugger/.fastai/data/biwi_head_pose/01.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/18.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/04'),Path('/home/sgugger/.fastai/data/biwi_head_pose/10.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/24'),Path('/home/sgugger/.fastai/data/biwi_head_pose/14.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/20.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/11.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/02.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/07')...]
```
24개의 주소가 01부터 24까지로 구별되어 있고 (각각 다른 사람의 사진으로 구별) 그에 따른 .obj 파일도 있다 (여기에서는 사용되지 않는다). 이 주소들 중 하나를 살펴보도록 하겠다:
```
(path/'01').ls()
```
```
(#1000) [Path('01/frame_00087_pose.txt'),Path('01/frame_00079_pose.txt'),Path('01/frame_00114_pose.txt'),Path('01/frame_00084_rgb.jpg'),Path('01/frame_00433_pose.txt'),Path('01/frame_00323_rgb.jpg'),Path('01/frame_00428_rgb.jpg'),Path('01/frame_00373_pose.txt'),Path('01/frame_00188_rgb.jpg'),Path('01/frame_00354_rgb.jpg')...]
```
하위 디렉토리 내부에는, 각기 다른 프레임과, 프레임 별로 이미지(\_rgb.jpg)와 포즈 파일(\_pose.txt)이 함께 제공됩니다. [get_image_files](https://docs.fast.ai/data.transforms.html#get_image_files) 을 사용하여 모든 이미지 파일을 재귀적으로 쉽게 가져온 다음 이미지 파일 이름을 연결된 포즈 파일로 변환하는 함수를 작성할 수 있습니다.
```
img_files = get_image_files(path)
def img2pose(x): return Path(f'{str(x)[:-7]}pose.txt')
img2pose(img_files[0])
```
```
Path('04/frame_00084_pose.txt')
```
첫 번째 이미지 확인 방법:
```
im = PILImage.create(img_files[0])
im.shape
```
```
(480, 640)
```
```
im.to_thumb(160)
```
 
Biwi 데이터 셋 웹사이트에서는 중심점의 위치를 보여주는 각 이미지와 관련된 포즈 텍스트 파일의 형식을 설명해주고 있습니다. 이것의 세부 사항은 우리의 목적에 중요하지 않으므로 중심점을 추출하는데 사용되는 함수를 사용해야 합니다:
```
cal = np.genfromtxt(path/'01'/'rgb.cal', skip_footer=6)
def get_ctr(f):
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0]/ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1]/ctr[2] + cal[1][2]
    return tensor([c1,c2])
```
이 함수는 좌표를 두 항목의 텐서로 반환합니다:
```
get_ctr(img_files[0])
```
```
tensor([372.4046, 245.8602])
```
이 함수는 각 항목에 레이블을 지정하는 역할을 하기 [DataBlock](https://docs.fast.ai/data.block.html#DataBlock) 에 get_y로 전달할 수 있습니다. 또한 훈련 속도를 조금 높이기 위해 이미지 크기를 입력 크기의 절반으로 조정하겠습니다.

한 가지 주의할 점은 무작위로 흩어서는 안 된다는 것입니다. 그 이유는 동일한 사람이 데이터 셋의 여러 이미지에 나타나기 때문인데—하지만 우리는 모델을 아직 이미지를 본 적 없는 사람들에게도 일반화해서 적용하고 싶기 때문이다. 데이터 셋의 각 폴더에는 한 사람의 이미지가 포함되어 있습니다. 따라서 한 사람에 대해서만 true를 반환하는 splitter함수를 생성하여 그 사람의 이미지만 포함하는 유효성 검사 셋을 생성할 수 있습니다.

이전 데이터 블록 예제와 다른 유일한 차이점은 두번째 블록이 PointBlock 이라는 점입니다. 이것은 fastai가 레이블이 좌표를 나타낸다는 것을 알기 위해서 필요하기 때문에 데이터 증가를 수행할 때 이미지에 대해 수행하는 것처럼 이러한 좌표에 대해 동일한 증가를 수행해야 한다는 것을 알려줍니다.
```
biwi = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_ctr,
    splitter=FuncSplitter(lambda o: o.parent.name=='13'),
    batch_tfms=[*aug_transforms(size=(240,320)), 
                Normalize.from_stats(*imagenet_stats)]
)
dls = biwi.dataloaders(path)
dls.show_batch(max_n=9, figsize=(8,6))
```

이렇게 데이터들을 다 모으면, 이제 나머지 fastai API를 사용할 수 있습니다. [cnn_learner](https://docs.fast.ai/vision.learner.html#cnn_learner) 는 이 경우에 완벽히 작동하며, 라이브러리는 데이터에서 적절한 손실 함수를 추론하게 됩니다:
```
learn = cnn_learner(dls, resnet18, y_range=(-1,1))
learn.lr_find()
```
```
(0.005754399299621582, 3.6307804407442745e-07)
```
 
이제 모델을 학습시킬 수 있습니다:
```
learn.fine_tune(1, 5e-3)
```
epoch	train_loss	valid_loss	time
0	0.057434	0.002171	00:31
epoch	train_loss	valid_loss	time
0	0.005320	0.005426	00:39
1	0.003624	0.000698	00:39
2	0.002163	0.000099	00:39
3	0.001325	0.000233	00:39
손실은 평균 제곱 오차이므로 포인트를 예측할 때 평균적으로 퍼센트 오차가 발생합니다.
```
math.sqrt(0.0001)
```
```
0.01
```
최종적인 결과입니다:
```
learn.show_results()
```

