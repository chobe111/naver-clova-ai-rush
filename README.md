# 1-3 네이버 클라우드 마이박스의 이미지 분류

네이버 클라우드 마이박스의 이미지 분류 테마를 11개에서 40여개로 확대 적용합니다.

# Introduction

이 문제는 실제 마이박스에서 수집된 이미지 테마들을 분류하는 문제입니다. 이 문제의 특성은 아래와 같습니다.
(1) Few-shot classification 문제입니다.
(2) 각 이미지 별로 multi-class 형태의 label을 지니고 있습니다. 이 class는 hierarchy를 띄고 있습니다.
[Image TBA]
이 이미지는 A_B 카테고리로 분류되어 있으며 이 때 label은 A와 B입니다.

# Dataset Detail
- Dataset ID/개수:
- Dataset 구성:
- Label 구성: 기본적으로 ID 개수 만큼의 dimension을 가지는 one-hot vector로 구성이 됩니다. 예를 들어 그림 1의 label은 이 될 것입니다.
- Number of given Few-shot Images:
- Pretrained Dataset: 본 문제에서는 Few-shot으로 주어지는 이미지 외에 ImageNet 데이터셋을 추가로 활용하는 것이 가능합니다.
- Full Data Hierarchy

# Code spec
- 이미지 한장을 입력으로 받아 label id dimension의 binary one-hot vector를 출력.

# Measuring

본 문제에서는 실제 application에서 활용을 하기 위한 특화된 measuring을 제공합니다. 각 이미지는 0~1 사이의 실수 값으로 measuring 됩니다.
예를 들어 A-B-C 클래스로 labeling 된 이미지가 있다면

(1) 전체 클래스를 모두 맞춘 경우에는 최고 점수입니다 (A,B,C). 이 경우에는 각 이미지당 1 점으로 판정됩니다.

(2) 상위 클래스를 맞춘 경우에는 부분점수가 있습니다. 부분점수는 맞춘 클래스 개수 / gt class의 개수 가 됩니다.

- (A-B, A-C, B-C) -> 이 경우 2/3 점 입니다.
- (A, B, C) -> 이 경우에는 1/3 점이 됩니다.

Note: 점수는 중첨되지는 않습니다.

(3) 전체 결과는 총 test image 점수의 평균으로 판정됩니다.

# Requirements and warning

(0) 참가는 PyTorch (>=1.6, Including TorchVision >= 0.7.0) 으로 진행됩니다.

(1) ImageNet dataset을 사용하기 위해서는 첫째, AIRUSH NSML에 업로드 된 이미지넷 데이터셋을 직접 사용, 두번째로는 pytorch에서 사용되는 official pretrained code 사용, 두 가지 방법이 있습니다. 제출하시는 코드는 운영진에서 검수 가능하니 유의하시기 바랍니다.

(2) model 은 32bit FP 모델을 사용해 주시기 바랍니다. 이미지 사이즈는 제한이 없습니다.

(3) 이미지 크기의 제한은 걸려있지 않지만 model의 FLops 제한이 걸려있습니다. 32bit FP 모델 기준 2GFlops를 초과하지 않도록 해주세요. 이 이상 크기를 가지는 모델은 Reject될 예정입니다.

(4) Flops 계산은 https://github.com/sovrasov/flops-counter.pytorch 를 기준으로 진행될 예정입니다.


# ETC

더 자세한 내용은 example code를 참조해 주세요.
