# [Track 2] 2023 바이오헬스 데이터 경진대회 - 치의학 분야 (일반 부문)
- 대회기간 : 2023-11-20 ~ 2023-11-27
- Team : 김다현, 최희영, 지경호, 양재하
<br/>
<p align="center">
<img src="https://github.com/MrSteveChoi/AI_projects/assets/132117793/8795d664-fc2d-43d0-ac44-8c0863a97bbc" width=50% height=50%>
</p>
<br/>

## 대회 목표 및 과제 <br/>
 - 목표 및 기대효과 : 바이오헬스 데이터와 AI 기술을 활용하여 치의학 분야에서의 위험성 평가와 예측 모델 개발
 - 과제 : 사랑니 이미지와 환자 데이터를 바탕으로 사랑니 발치 시간을 예측하는 모델 개발
---

## 대회 진행 과정 <br/>
### 1. Data set
Data set | 개수  
:---: | :---: | 
train images | 432
test images | 286
meta data | -

- Data set 예상도 및 예시 (**image 파일 시각화 불가능**)
 - image 파일 예상도 <br> 
   - Panorama X-ray에서 하악 사랑니를 crop한 image <br>
   - 사랑니 앞치아와의 관계 및 하치조신경이 포함된 사진일 것으로 예상 <br>
  <img src="https://github.com/MrSteveChoi/AI_projects_Balchi/assets/132117793/727e1870-d8f1-432d-9a0c-c5c7f9305f4a" width=70% > <br>
 - meta data 예시 <br>
  <img src="https://github.com/MrSteveChoi/AI_projects_Balchi/assets/132117793/344550b8-a33a-477c-ada1-aad65826d951" width=70% > <br>

<br>

### 2. 훈련 데이터 명세
 Image Augmentation을 통해 증강한 총 2,500개 데이터를 학습용 데이터(Training Set), 검증 및 모델 선택용 데이터(Validation Set)로 나누었고, 그 비율은 8:2로 설정했습니다.
Num total | TrainSet | ValidSet
:---: | :---: | :---: |
2,500 | 2,000(80%) | 500(20%)
<br/>

### 3. 모델 학습 과정
<img src="https://github.com/MrSteveChoi/AI_projects_Balchi/assets/132117793/4e169f13-3a53-435a-8564-91c31d126027" width=70% > <br/>
<br/>

### 4. 결과
- Metric: MAPE <br/>
- Score
  - Final Score : 9th / 0.4422 <br/>
  - Public Score : 9th / 0.5525 <br/>
  - Private Score: 9th / 0.3949 <br/>
  - https://aiconnect.kr/competition/detail/233/task/307/leaderboard <br/>

---

### 회고
- 아쉬운 점
    - 약 일주일 정도의 짧은 대회 기간이였기 때문에, 대회 시작 전에 CLI환경에서 사용 가능한 코드를 작성해 시간을 조금 더 효율적으로 사용했으면 어땠을까 하는 아쉬움이 남습니다.
    - 대회에서 제공받은 환경에 computing resource의 한계로 ViT와 같은 큰 모델을 적극적으로 사용하지 못한 것이 아쉬웠습니다.
    - 전체 제출 횟수가 부족하여 해보고 싶은 실험을 모두 할 수 없었던 점이 아쉬웠습니다.
    - 시술자의 실력은 날이 갈수록 좋아지기 때문에 선형적으로 시술 시간이 줄어드는 것을 확인하였으나 이를 적극 활용하지 못했습니다. 해당 대회에서 시술자를 기준으로 모델을 따로 학습시켜 inference하는 방식을 적용해봤으면 좋았을 것 같습니다.
    - time feature를 활용하여 첫 날짜를 기준으로 시간이 지날수록 해당 의사에게 가중치를 주는 방식으로 예측하는 방법을 적용해 보았으면 좋았을 것 같습니다.
    - 발치 시간을 예측하는 “회귀” 문제를 시간대 별로 binning하여 다중 분류 문제로 접근해 보면 어땠을까 하는 아쉬움이 있습니다.
- 느낀 점
    - 폐쇄 환경에서 CLI만을 이용해서 end-to-end로 학습을 해 보는 좋은 기회가 되었습니다.
    - CLI환경에서만 학습을 해야 했기 때문에 함수들의 모듈화를 시도해볼 수 있는 좋은 기회였습니다.
    - 이미지와 메타데이터 모두를 한 번에 사용하는 multi-modal training을 직접 적용해볼 수 있었습니다.
    - 순위권 팀들의 성과공유를 통해 학습에 적용한 기술적인 방법론이 비슷하다 하더라도 작은 디테일들에 의해 성능이 크게 달라질 수 있다는 것을 느꼈습니다.
    - doctor와 patient간의 attention score와 이미지의 feature를 합친 custom model을 직접 설계해서 사용해볼 수 있겠다는 생각을 하였습니다.
- 얻은 것
    - 제한된 제출기회를 의미있게 활용하려면 public score와 상관관계를 가지는 내부적인 evaluation metric을 빠르게 확보하는 것이 중요하다는 점을 느꼈습니다.
    - 대회에서 사용할 모델을 선택하기 전에 관련 대회나 논문을 살펴보고 결과가 좋았던 모델을 선택하는 방법이 있다는 것을 배웠습니다.
    - 정해진 loss-metric이외에도 해당 task에 어울리는 다른 loss를 적용해 모델을 학습하는 것이 모델의 학습 결과에 큰 영향을 끼친다는 것을 배웠습니다.
- 다음에 할 것
    - 1등 팀이 공개한 attention을 활용한 custom모델을 참고해서 직접 구현해 보려고 합니다.
<br/>

---

### 기술 스택

- python  <br/>
- pytorch  <br/>
- albumentation  <br/>

### Reference

---
### 주관 / 주최
 - 주최 : 홍익대 바이오헬스 혁신융합대학사업단, 삼성서울병원, 경희대학교치과병원 <br/>
 - 주관 : AI CONNECT <br/>
<br/>

### Vim 환경설정

- vim plugin 설치 할 때 참고한 사이트
  - [Vim plugin 관리를 위한 번들(Vundle)](https://github.com/VundleVim/Vundle.vim)
  
  - [Jedi for vim autocompletion](https://vimawesome.com/plugin/jedi-vim)
  
  - [Flake8 for vim syntax & style checker](https://vimawesome.com/plugin/vim-flake8)
  <br>

1. Vundle 설치아래 명령어 실행

```
git clonehttps://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
```
<br>

2. `~/.vimrc`에 아래 탬플릿 작성(큰따옴표로 주석처리)

```
set nocompatible              " be iMproved, required 
filetype off                  " required 

set rtp+=~/.vim/bundle/Vundle.vim 
call vundle#begin() 

" 여기에 플러그인 설치 사용법에 따라 추가작성 

call vundle#end()            " required 
filetype plugin indent on    " required
```
<br>

3. ```~/.vimrc``` 파일에 다음 명령어 추가

```
Plugin 'davidhalter/jedi-vim'
Plugin 'nvie/vim-flake8'
```
<br>
4. vim에서 다음 명령어 실행

```
:source %
:PluginInstall
```
