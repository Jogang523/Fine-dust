# 백엔드
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import requests
from pydantic import BaseModel
from typing import List
import uvicorn
import chardet
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base 
from pydantic import BaseModel
from sqlalchemy.orm import Session
import xml.etree.ElementTree as ET
import logging
import joblib
import folium
import geopandas as gpd
import matplotlib.font_manager as fm
import matplotlib as mpl
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
from datetime import datetime, timedelta


app = FastAPI()

#진자템플릿
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
templates = Jinja2Templates(directory="templates")
templates.env.globals['max'] = max
templates.env.globals['min'] = min

# 사용자 정의 mse 함수
def dummy_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# 사용자 정의 객체 딕셔너리
custom_objects = {'mse': dummy_mse}


# db ----------------------------------------------------------------------------
database_url = "mysql+pymysql://root:1234@localhost/shelter_db"
engine = create_engine(database_url)
Base = declarative_base()

# 테이블 생성
class Shelter(Base):
    __tablename__ = "shelters"  # 테이블 이름

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    district_name = Column(String(100), index=True)
    facility_name = Column(String(200), unique=True, index=True)
    address = Column(String(255))
    facility_type = Column(String(100))
    capacity = Column(Integer)
    tel = Column(String(20))
    weekday_hours = Column(String(100))
    x = Column(String(100))
    y = Column(String(100))
    
# pydantic model 
class ShelterCreate(BaseModel):
    district_name: str
    facility_name: str
    address: str
    facility_type: str
    capacity: int
    tel: str
    weekday_hours: str
    latitude: float
    longtitude: float

# database session 생성하고 관리하는 함수
def get_db():
    db = Session(bind=engine)  # db와 연결
    try:
        yield db
    finally:
        db.close()

Base.metadata.create_all(bind=engine)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AirQualityData(BaseModel):
    MSRDTE: str
    MSRRGN_NM: str
    MSRSTE_NM: str
    PM10: float
    PM25: float
    O3: float
    NO2: float
    CO: float
    SO2: float

# 메인 페이지 ---------------------------------------------------------------------------------------

# 미세먼지 실시간 api 지도
def create_map(df, geo_data):

    m = folium.Map(location=[37.5665, 126.9780], zoom_start=10.3)

    bins = [0, 30, 80, 150, np.inf]
    colors = ['#89CFF0', '#32CD32', '#FFD700', '#FF4500']

    def color_producer(value):
        if pd.isna(value):
            return 'gray'
        elif value <= bins[1]:
            return colors[0]
        elif value <= bins[2]:
            return colors[1]
        elif value <= bins[3]:
            return colors[2]
        else:
            return colors[3]
        
    
    for _, row in geo_data.iterrows():
        value = df[df['MSRSTE_NM'] == row['SIG_KOR_NM']]['미세먼지'].mean()
        folium.GeoJson(
            row['geometry'],
            style_function=lambda x, value=value: {
                'fillColor': color_producer(value),
                'color': 'gray',
                'weight': 1,
                'fillOpacity': 0.7,
            },
            popup=folium.Popup(f"미세먼지: {value}", max_width=100)
        ).add_to(m)



    # 오른쪽 상단에 시간 정보 추가
    time_info = df['MSRDT'].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
    folium.map.Marker(
        location=[37.7, 127.2],
        icon=folium.DivIcon(html=f"""
            <div style="font-family: NanumSquare; font-size: 12pt; color: black; background-color: white; padding: 5px; border: 1px solid black; white-space: nowrap; display: inline-block; font-weight: bold;">
                {time_info}
            </div>
        """)
    ).add_to(m)



    # 자치구 이름 표시
    for _, row in df.iterrows():
        center = geo_data[geo_data['SIG_KOR_NM'] == row['MSRSTE_NM']].geometry.centroid.iloc[0]
        folium.Marker(
            location=[center.y, center.x],
            icon=folium.DivIcon(html=f"""
                <div style="font-family: NanumSquare; font-size: 7pt; color: black; text-align: center; white-space: nowrap; font-weight: bold;">
                {row['MSRSTE_NM']}
                </div>
            """)
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m

#-----------------------------------------------------------------------
# 초미세먼지
def create_s_map(df, geo_data):

    s_m = folium.Map(location=[37.5665, 126.9780], zoom_start=10.3)

    bins = [0, 15, 30, 75, np.inf]
    colors = ['#89CFF0', '#32CD32', '#FFD700', '#FF4500']

    def color_producer(value):
        if pd.isna(value):
            return 'gray'
        elif value <= bins[1]:
            return colors[0]
        elif value <= bins[2]:
            return colors[1]
        elif value <= bins[3]:
            return colors[2]
        else:
            return colors[3]
        
    
    for _, row in geo_data.iterrows():
        value = df[df['MSRSTE_NM'] == row['SIG_KOR_NM']]['초미세먼지'].mean()
        folium.GeoJson(
            row['geometry'],
            style_function=lambda x, value=value: {
                'fillColor': color_producer(value),
                'color': 'gray',
                'weight': 1,
                'fillOpacity': 0.7,
            },
            popup=folium.Popup(f"초미세먼지: {value}", max_width=100)
        ).add_to(s_m)



    # 오른쪽 상단에 시간 정보 추가
    time_info = df['MSRDT'].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
    folium.map.Marker(
        location=[37.7, 127.2],
        icon=folium.DivIcon(html=f"""
            <div style="font-family: NanumSquare; font-size: 12pt; color: black; background-color: white; padding: 5px; border: 1px solid black; white-space: nowrap; display: inline-block; font-weight: bold;">
                {time_info}
            </div>
        """)
    ).add_to(s_m)



    # 자치구 이름 표시
    for _, row in df.iterrows():
        center = geo_data[geo_data['SIG_KOR_NM'] == row['MSRSTE_NM']].geometry.centroid.iloc[0]
        folium.Marker(
            location=[center.y, center.x],
            icon=folium.DivIcon(html=f"""
                <div style="font-family: NanumSquare; font-size: 7pt; color: black; text-align: center; white-space: nowrap; font-weight: bold;">
                {row['MSRSTE_NM']}
                </div>
            """)
        ).add_to(s_m)

    folium.LayerControl().add_to(s_m)
    return s_m 

def get_real_time_data():
    url = 'http://openAPI.seoul.go.kr:8088/6d6d524c626869393635704d4a5667/json/RealtimeCityAir/1/30/'
    res = requests.get(url)
    data = res.json()

    RealtimeData = data['RealtimeCityAir']['row']
    df = pd.DataFrame(RealtimeData)
    
    df['MSRDT'] = pd.to_datetime(df['MSRDT'], format='%Y%m%d%H%M')

    numeric_columns = ['PM10', 'PM25', 'O3', 'NO2', 'CO', 'SO2']
    
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    gdf = pd.read_csv(r'.\data\서울시_자치구_경위도.csv')

    for index, value in enumerate(gdf['field1']):
        gdf.at[index, 'field1'] = value.replace('청', '')
    
    # df = df.dropna(subset=['PM10'])

    gdf.rename(columns={'field1': 'MSRSTE_NM'}, inplace=True)
        
    merged_data = pd.merge(df, gdf, on='MSRSTE_NM')
    merged_data.rename(columns={'PM10': '미세먼지','PM25': '초미세먼지'}, inplace=True)    
    return merged_data


@app.get("/")
async def home(request: Request):
    df = get_real_time_data()
    
    # 지도 데이터 불러오기
    geo_data = gpd.read_file(r'./data/서울_자치구_경계_2017.geojson') ##@@@@@@@@@경로
    
    # 지도 생성
    m = create_map(df, geo_data)
    s_m = create_s_map(df, geo_data)
    # 지도를 HTML 문자열로 변환
    map_html = m._repr_html_()
    s_map_html = s_m._repr_html_()
    return templates.TemplateResponse('index2.html', {"request": request, "map": map_html, "s_map":s_map_html})



# 미세먼지 예측 ---------------------------------------------------------------------------------------

# 모델 로드
model_pm10 = load_model('./model/lstm_model_pm10.h5', compile=False)
model_pm25 = load_model('./model/lstm_model_pm25.h5', compile=False)

scaler = joblib.load('./model/scaler.joblib')


def fetch_data():
    url = 'http://openAPI.seoul.go.kr:8088/6d6d524c626869393635704d4a5667/json/RealtimeCityAir/1/30/'
    res = requests.get(url)
    data = res.json()
    return pd.DataFrame(data['RealtimeCityAir']['row'])

def preprocess_data(df):
    df['MSRDT'] = pd.to_datetime(df['MSRDT'], format='%Y%m%d%H%M')
    numeric_columns = ['NO2', 'O3', 'CO', 'SO2', 'PM10', 'PM25']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df['Month'] = df['MSRDT'].dt.month
    df['Day'] = df['MSRDT'].dt.day
    df['Hour'] = df['MSRDT'].dt.hour
    return df

def prepare_lstm_input(df):
    input_data = df[['NO2', 'O3', 'CO', 'SO2', 'PM10', 'PM25', 'Month', 'Day', 'Hour']].values
    scaled_data = scaler.transform(input_data)
    return np.expand_dims(scaled_data[-30:], axis=0)  # 최근 30개 데이터 사용

def predict_next_6_days(model, input_data):
    predictions = []
    current_input = input_data.copy()
    
    for _ in range(6 * 24):  # 6일 * 24시간
        next_hour_pred = model.predict(current_input, verbose=0)[0, 0]
        predictions.append(next_hour_pred)
        
        # 다음 예측을 위해 입력 데이터 업데이트
        new_row = current_input[0, -1, :].copy()
        new_row[4] = next_hour_pred  # PM10 또는 PM25 업데이트
        new_row[6] = (new_row[6] % 12) + 1  # Month
        new_row[7] = (new_row[7] % 30) + 1  # Day (간단히 처리)
        new_row[8] = (new_row[8] + 1) % 24  # Hour
        
        current_input = np.roll(current_input, -1, axis=1)
        current_input[0, -1, :] = new_row
    
    return predictions

@app.get("/chart")
async def chart(request: Request):
    try:
        df = fetch_data()
        processed_df = preprocess_data(df)
        
        lstm_input = prepare_lstm_input(processed_df)
        
        pm10_predictions = predict_next_6_days(model_pm10, lstm_input)
        pm25_predictions = predict_next_6_days(model_pm25, lstm_input)
        
        # 현재 평균값 계산
        current_pm10_avg = processed_df['PM10'].mean()
        current_pm25_avg = processed_df['PM25'].mean()
        
        # 예측값 후처리
        pm10_predictions = [round(max(0, min(p, 200)), 2) for p in pm10_predictions]
        pm25_predictions = [round(max(0, min(p, 200)), 2) for p in pm25_predictions]
        
        # 날짜 생성
        dates = [(datetime.now() + timedelta(hours=i)).strftime("%Y-%m-%d %H:00") for i in range(len(pm10_predictions))]
        
        return templates.TemplateResponse("chart.html", {
            "request": request,
            "current_pm10": round(current_pm10_avg, 2),
            "current_pm25": round(current_pm25_avg, 2),
            "pm10_predictions": pm10_predictions,
            "pm25_predictions": pm25_predictions,
            "dates": dates
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 미세먼지 대피소 -------------------------------------------------------------------------------------

@app.get("/form")
async def form(request: Request, page: int = 1):
    # 전체 데이터를 저장할 데이터프레임 초기화
    df = pd.DataFrame(columns=['id', 'district_name', 'facility_name', 'address', 'facility_type', 'capacity', 'tel', 'weekday_hours', 'latitude', 'longitude'])
    
    # 페이지를 넘기며 데이터를 가져오기
    current_page = 1
    while True:
        # API 가져오기
        start_index = (current_page - 1) * 10 + 1
        end_index = current_page * 10
        url = f'http://openapi.seoul.go.kr:8088/6d6d524c626869393635704d4a5667/xml/shuntPlace/{start_index}/{end_index}/'
        res = requests.get(url)
        root = ET.fromstring(res.content)
        
        # XML에서 데이터 추출 및 데이터프레임에 추가
        items = root.findall('.//row')
        if not items:
            break  # 더 이상 데이터가 없으면 종료
        
        for i, item in enumerate(items, start=start_index):
            district_name = item.find('HJD_NAM').text if item.find('HJD_NAM') is not None else "Unknown"
            facility_name = item.find('SHUNT_NAM').text if item.find('SHUNT_NAM') is not None else "Unknown"
            address = item.find('ADR_NAM').text if item.find('ADR_NAM') is not None else "Unknown"
            facility_type = item.find('EQUP_TYPE').text if item.find('EQUP_TYPE') is not None else "Unknown"
            capacity = item.find('HOU_CNT_M').text if item.find('HOU_CNT_M') is not None else "Unknown"
            tel = item.find('TEL_NO_CN').text if item.find('TEL_NO_CN') is not None else "Unknown"
            weekday_hours = item.find('WKDY_USE_HR').text if item.find('WKDY_USE_HR') is not None else "Unknown"
            latitude = item.find('MAP_COORD_X').text if item.find('MAP_COORD_X') is not None else "0.0"
            longitude = item.find('MAP_COORD_Y').text if item.find('MAP_COORD_Y') is not None else "0.0"
            
            # 데이터프레임에 행 추가
            df = pd.concat([df, pd.DataFrame({
                'id': [i],
                'district_name': [district_name],
                'facility_name': [facility_name],
                'address': [address],
                'facility_type': [facility_type],
                'capacity': [capacity],
                'tel': [tel],
                'weekday_hours': [weekday_hours],
                'latitude': [latitude],
                'longitude': [longitude]
            })], ignore_index=True)
        
        current_page += 1
    
    # 총 데이터 수와 페이지 수 계산
    total_rows = len(df)
    total_pages = (total_rows // 10) + (1 if total_rows % 10 > 0 else 0)
    
    # 요청한 페이지에 해당하는 데이터만 추출
    start_row = (page - 1) * 10
    end_row = start_row + 10
    page_data = df.iloc[start_row:end_row].to_dict('records')
    
    return templates.TemplateResponse('form.html', {
        "request": request, 
        "shelters": page_data, 
        "current_page": page, 
        "total_pages": total_pages
    })
