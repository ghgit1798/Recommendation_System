from django.shortcuts import render

import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# 우리가 예측한 평점과 실제 평점간의 차이를 MSE로 계산
def get_mse(pred, actual):
    # 평점이 있는 실제 영화만 추출
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


# 특정 맥주와 비슷한 유사도를 가지는 맥주 Top_N에 대해서만 적용 -> 시간오래걸림
def predict_rating_topsim(ratings_arr, item_sim_arr, n=20):
    # 사용자-아이템 평점 행렬 크기만큼 0으로 채운 예측 행렬 초기화
    pred = np.zeros(ratings_arr.shape)

    # 사용자-아이템 평점 행렬의 맥주 개수만큼 루프
    for col in range(ratings_arr.shape[1]):
        # 유사도 행렬에서 유사도가 큰 순으로 n개의 데이터 행렬의 인덱스 반환
        top_n_items = [np.argsort(item_sim_arr[:, col])[:-n - 1:-1]]
        # 개인화된 예측 평점 계산 : 각 col 맥주별(1개), 2496 사용자들의 예측평점
        for row in range(ratings_arr.shape[0]):
            pred[row, col] = item_sim_arr[col, :][top_n_items].dot(
                ratings_arr[row, :][top_n_items].T)
            pred[row, col] /= np.sum(item_sim_arr[col, :][top_n_items])

    return pred

# 사용자가 안 먹어본 맥주를 추천하자.
def get_not_tried_beer(ratings_matrix, userId):
    # userId로 입력받은 사용자의 모든 맥주 정보를 추출해 Series로 반환
    # 반환된 user_rating은 영화명(title)을 인덱스로 가지는 Series 객체
    user_rating = ratings_matrix.loc[userId, :]

    # user_rating이 0보다 크면 기존에 관란함 영화.
    # 대상 인덱스를 추출해 list 객체로 만듦
    tried = user_rating[user_rating > 0].index.tolist()

    # 모든 맥주명을 list 객체로 만듦
    beer_list = ratings_matrix.columns.tolist()

    # list comprehension으로 tried에 해당하는 영화는 beer_list에서 제외
    not_tried = [beer for beer in beer_list if beer not in tried]

    return not_tried

# 예측 평점 DataFrame에서 사용자 id 인덱스와 not_tried로 들어온 맥주명 추출 후
# 가장 예측 평점이 높은 순으로 정렬
def recomm_beer_by_userid(pred_df, userId, not_tried, top_n):
    recomm_beer = pred_df.loc[userId, not_tried].sort_values(ascending=False)[:top_n]
    return recomm_beer

# 평점, Aroma, Flavor, Mouthfeel 중 피처 선택 후 유사도 계산
def recomm_feature(df, col):
    feature = col
    ratings = df[['아이디','맥주', feature]]

    # 피벗 테이블을 이용해 유저-아이디 매트릭스 구성
    ratings_matrix = ratings.pivot_table(feature, index='아이디', columns='맥주')
    ratings_matrix.head(3)
    # fillna함수를 이용해 Nan처리
    ratings_matrix = ratings_matrix.fillna(0)

    # 유사도 계산을 위해 트랜스포즈
    ratings_matrix_T = ratings_matrix.transpose()

    # 아이템-유저 매트릭스로부터 코사인 유사도 구하기
    item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)

    # cosine_similarity()로 반환된 넘파이 행렬에 영화명을 매핑해 DataFrame으로 변환
    item_sim_df = pd.DataFrame(data=item_sim, index=ratings_matrix.columns,
                              columns=ratings_matrix.columns)

    return item_sim_df

# 해당 맥주와 유사한 유사도 5개 추천
def recomm_beer(item_sim_df, beer_name):
    # 해당 맥주와 유사도가 높은 맥주 5개만 추천
    return item_sim_df[beer_name].sort_values(ascending=False)[1:4]

def index(request):
    return render(request, 'beer/index.html')

def ver1(request):
    beer_list = pd.read_csv('맥주이름.csv', encoding='utf-8', index_col=0)
    beer_year = pd.read_csv('맥주_연도별평점.csv', encoding='utf-8', index_col=0)
    ratings = pd.read_csv('정제된데이터.csv', encoding='utf-8', index_col=0)
    cluster_3 = pd.read_csv('대표군집클러스터링.csv', encoding='utf-8', index_col=0)
    cluster_all = pd.read_csv('전체맥주클러스터링.csv', encoding='utf-8', index_col=0)
    beer_data = pd.read_csv('맥주_cbf_data.csv', encoding='utf-8', index_col=0)
    beer_list = beer_list['맥주']
    cluster_3 = cluster_3.values

    if request.method == 'POST':
        beer_name = request.POST.get('beer', '')
        detail = request.POST.get('detail', '')
        df_aroma = recomm_feature(ratings, 'Aroma')
        df_flavor = recomm_feature(ratings, 'Flavor')
        df_mouthfeel = recomm_feature(ratings, 'Mouthfeel')

        if detail=='Aroma':
            df = df_aroma*0.8 + df_flavor*0.1 + df_mouthfeel*0.1
        if detail=='Flavor':
            df = df_aroma*0.1 + df_flavor*0.8 + df_mouthfeel*0.1
        if detail=='Mouthfeel':
            df = df_aroma*0.1 + df_flavor*0.1 + df_mouthfeel*0.8

        result = recomm_beer(df, beer_name)
        result = result.index.tolist()

        # 클러스터링 결과
        tmp_cluster=[]
        category=[]
        food=[]
        for i in range(3):
            target = cluster_all[cluster_all['맥주']==result[i]]
            target = target[['Aroma', 'Appearance', 'Flavor', 'Mouthfeel', 'Overall']]
            target = target.values[0]
            tmp_cluster.append(target)

            try :
                category.append(beer_data[beer_data['맥주이름']==result[i]]['Main Category'].values[0])
                food.append(beer_data[beer_data['맥주이름']==result[i]]['Paring Food'].values[0])
            except :
                category.append('수집되지 않았습니다.')
                food.append('수집되지 않았습니다.')

        # 연도별 평점 결과
        tmp_year = []
        tmp_ratings = []
        for i in range(3):
            target = beer_year[beer_year['맥주']==result[i]]
            target_year = target['년'].tolist()
            target_rating = target['평점'].tolist()
            tmp_year.append(target_year)
            tmp_ratings.append(target_rating)

        # 넘겨줄 데이터 Json 변환
        targetdict = {
            'beer_name' : result,
            'beer_cluster1' : tmp_cluster[0].tolist(),
            'beer_cluster2' : tmp_cluster[1].tolist(),
            'beer_cluster3' : tmp_cluster[2].tolist(),
            'cluster1' : cluster_3[0].tolist(),
            'cluster2' : cluster_3[1].tolist(),
            'cluster3' : cluster_3[2].tolist(),
            'tmp_year' : tmp_year,
            'tmp_ratings' : tmp_ratings
        }

        targetJson = json.dumps(targetdict)

        return render(request, 'beer/ver1_result.html',
                   {'result':result, 'beer_list':beer_list,'targetJson':targetJson,
                    'category':category, 'food':food})
    else:
        return render(request, 'beer/ver1.html', {'beer_list':beer_list})

def ver2(request):
    beer_list = pd.read_csv('맥주이름.csv', encoding='utf-8', index_col=0)
    beer_year = pd.read_csv('맥주_연도별평점.csv', encoding='utf-8', index_col=0)
    ratings = pd.read_csv('정제된데이터.csv', encoding='utf-8', index_col=0)
    cluster_3 = pd.read_csv('대표군집클러스터링.csv', encoding='utf-8', index_col=0)
    cluster_all = pd.read_csv('전체맥주클러스터링.csv', encoding='utf-8', index_col=0)
    beer_data = pd.read_csv('맥주_cbf_data.csv', encoding='utf-8', index_col=0)
    beer_list = beer_list['맥주']
    cluster_3 = cluster_3.values

    if request.method == 'POST':
        name = request.POST.get('name', '')
        beer = []
        rating = []
        for i in range(1,6):
            beer.append(request.POST.get('beer'+str(i), ''))
            rating.append((request.POST.get('rating'+str(i), '')))

        for i in range(len(beer)):
            tmp = []
            tmp.append(name)
            tmp.append(beer[i])
            tmp.append(float(rating[i]))
            tmp = pd.DataFrame(data=[tmp], columns=['아이디','맥주','평점'])
            ratings = pd.concat([ratings, tmp])

        uname = name
        ratings_matrix = ratings.pivot_table('평점', index='아이디', columns='맥주')
        ratings_matrix = ratings_matrix.fillna(0)
        ratings_matrix_T = ratings_matrix.transpose()

        item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)
        item_sim_df = pd.DataFrame(data=item_sim, index=ratings_matrix.columns,
                                   columns=ratings_matrix.columns)

        # top_n과 비슷한 유저들만 추천에 사용
        ratings_pred = predict_rating_topsim(ratings_matrix.values, item_sim_df.values, n=5)
        # 계산된 예측 평점 데이터는 DataFrame으로 재생성
        ratings_pred_matrix = pd.DataFrame(data=ratings_pred, index=ratings_matrix.index,
                                           columns=ratings_matrix.columns)

        # 유저가 먹지 않은 맥주이름 추출
        not_tried = get_not_tried_beer(ratings_matrix, uname)
        # 아이템 기반의 최근접 이웃 CF로 맥주 추천
        recomm_beer = recomm_beer_by_userid(ratings_pred_matrix, uname, not_tried, top_n=3)
        recomm_beer = pd.DataFrame(data=recomm_beer.values, index=recomm_beer.index,
                                   columns=['예측평점'])
        # 추천 결과로 나온 맥주이름만 추출
        result = recomm_beer.index.tolist()

        # 클러스터링 결과
        tmp_cluster = []
        category = []
        food = []
        for i in range(3):
            target = cluster_all[cluster_all['맥주'] == result[i]]
            target = target[['Aroma', 'Appearance', 'Flavor', 'Mouthfeel', 'Overall']]
            target = target.values[0]
            tmp_cluster.append(target)

            try :
                category.append(beer_data[beer_data['맥주이름']==result[i]]['Main Category'].values[0])
                food.append(beer_data[beer_data['맥주이름']==result[i]]['Paring Food'].values[0])
            except :
                category.append('수집되지 않았습니다.')
                food.append('수집되지 않았습니다.')

        tmp_year = []
        tmp_ratings = []
        for i in range(3):
            target = beer_year[beer_year['맥주'] == result[i]]
            target_year = target['년'].tolist()
            target_rating = target['평점'].tolist()
            tmp_year.append(target_year)
            tmp_ratings.append(target_rating)

        targetdict = {
            'beer_name': result,
            'beer_cluster1': tmp_cluster[0].tolist(),
            'beer_cluster2': tmp_cluster[1].tolist(),
            'beer_cluster3': tmp_cluster[2].tolist(),
            'cluster1': cluster_3[0].tolist(),
            'cluster2': cluster_3[1].tolist(),
            'cluster3': cluster_3[2].tolist(),
            'tmp_year': tmp_year,
            'tmp_ratings': tmp_ratings
        }

        targetJson = json.dumps(targetdict)

        return render(request, 'beer/ver2_result.html',
                      {'result': result, 'beer_list': beer_list,'targetJson': targetJson,
                       'category':category, 'food':food})

    else:
        return render(request, 'beer/ver2.html', {'beer_list': beer_list})