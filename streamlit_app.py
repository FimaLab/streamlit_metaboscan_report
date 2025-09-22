import streamlit as st
import os
import tempfile
import pandas as pd
from datetime import datetime
import time
import requests
import numpy as np
import time
from functools import wraps
import json

# Add timing decorator
def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        st.session_state.setdefault('timing_data', {})[func.__name__] = execution_time
        return result
    return wrapper

# Import your utility functions (assuming they exist)
from ui_kit.streamlit_utilit import *

DASH_APP_URL = "https://metaboreport-test-rezvanov.amvera.io/"

# Add timing to your existing functions
@time_function
def clean_data_for_json(data):
    """Рекурсивно очищает данные от не-JSON-совместимых значений"""
    if isinstance(data, (np.floating, float)):
        if pd.isna(data) or np.isnan(data):
            return None
        elif np.isinf(data):
            return float('inf') if data > 0 else float('-inf')
        else:
            return float(data)
    
    elif isinstance(data, (np.integer, int)):
        return int(data)
    
    elif isinstance(data, (np.bool_, bool)):
        return bool(data)
    
    elif isinstance(data, (str, type(None))):
        return data
    
    elif isinstance(data, (list, tuple)):
        return [clean_data_for_json(item) for item in data]
    
    elif isinstance(data, dict):
        return {key: clean_data_for_json(value) for key, value in data.items()}
    
    elif isinstance(data, pd.DataFrame):
        return clean_data_for_json(data.to_dict('records'))
    
    elif isinstance(data, pd.Series):
        return clean_data_for_json(data.to_dict())
    
    else:
        try:
            return str(data)
        except:
            return None

@time_function
def dataframe_to_dict(df):
    """Convert DataFrame to JSON-serializable dict"""
    if df is None:
        return None
    return {
        'columns': list(df.columns),
        'data': clean_data_for_json(df.values.tolist()),
        'index': clean_data_for_json(df.index.tolist())
    }

@time_function
def update_dash_data(patient_info, data_dict):
    """Оптимизированная отправка данных в Dash с возвратом session_id"""
    try:
        # Prepare the data payload with DataFrames as serializable dicts
        payload = {
            'name': patient_info['name'],
            'age': patient_info['age'],
            'date': patient_info['date'],
            'gender': patient_info['gender'],
            'layout': patient_info['layout'],
            'metabolomic_data': dataframe_to_dict(data_dict['metabolomic_data'])
        }
        
        # Добавление опциональных сообщений
        for key in ['doctor_message', 'patient_message', 'patient_long_message']:
            if key in patient_info:
                payload[key] = patient_info[key]
        
        # Определяем URL для запроса
        current_session_id = st.session_state.get('session_id')
        if current_session_id:
            url = f"{DASH_APP_URL}/update_data/{current_session_id}"
        else:
            url = f"{DASH_APP_URL}/update_data"
        
        # Отправка запроса
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            # Получаем session_id из ответа
            response_data = response.json()
            session_id = response_data.get('session_id')
            
            # Сохраняем session_id в состоянии Streamlit
            if session_id:
                st.session_state.session_id = session_id
            
            return True, "Data updated successfully", session_id
        else:
            # Безопасное извлечение сообщения об ошибке
            try:
                error_data = response.json()
                error_msg = error_data.get('message', f'HTTP Error {response.status_code}')
            except (ValueError, TypeError):
                error_msg = response.text.strip() or f'HTTP Error {response.status_code}'
            return False, error_msg, None
        
    except Exception as e:
        error_msg = f"Ошибка при подготовке данных: {str(e)}"
        return False, error_msg, None

@time_function
def validate_inputs(name, file1):
    """Validate user inputs before processing"""
    if not name.strip():
        st.error("Please enter a valid patient name")
        return False
    if not file1:
        st.error("Please upload metabolomic data file")
        return False
    return True

@time_function
def download_pdf_from_dash(session_id):
    """Загрузка PDF из Dash приложения с использованием session_id"""
    try:
        response = requests.get(
            f"{DASH_APP_URL}/download_pdf/{session_id}",
            timeout=30,
        )
        
        if response.status_code == 200:
            pdf_data = response.content
            content_disposition = response.headers.get('Content-Disposition', '')
            filename = f"MetaboScan_Report_.pdf"
            
            if 'filename*=' in content_disposition:
                filename_part = content_disposition.split('filename*=')[1]
                if 'UTF-8\'\'' in filename_part:
                    filename = filename_part.split('UTF-8\'\'')[1].strip('";')
                else:
                    filename = filename_part.strip('";')
            elif 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[1].strip('"')
            
            return True, pdf_data, filename
        else:
            # Безопасное извлечение сообщения об ошибке
            try:
                error_data = response.json()
                error_msg = error_data.get('message', f'HTTP Error {response.status_code}')
            except (ValueError, TypeError):
                error_msg = response.text.strip() or f'HTTP Error {response.status_code}'
            return False, None, error_msg
            
    except Exception as e:
        return False, None, str(e)

@time_function
def check_dash_health():
    """Проверка доступности Dash приложения"""
    try:
        response = requests.get(f"{DASH_APP_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

@time_function
def generate_pdf_report_api(patient_info, data_dict):
    """Генерация PDF отчета через Dash API с session_id"""
    success, message, session_id = update_dash_data(patient_info, data_dict)
    
    if not success:
        st.error(f"Ошибка обновления данных: {message}")
        return None, None
    
    if not session_id:
        st.error("Не удалось получить session_id от сервера")
        return None, None
    
    # Сохраняем session_id для возможного повторного использования
    st.session_state.session_id = session_id
    
    time.sleep(1)  # Даем время на обработку данных
    
    with st.spinner("Генерация PDF отчета..."):
        success, pdf_data, result = download_pdf_from_dash(session_id)
        
        if success:
            return pdf_data, result
        else:
            st.error(f"Ошибка генерации PDF: {result}")
            return None, None

@time_function
def read_metabolomic_data(uploaded_file):
    """Read metabolomic data from uploaded file"""
    try:
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format for metabolomic data")
            return None
    except Exception as e:
        st.error(f"Error reading metabolomic data: {str(e)}")
        return None

def display_timing_report():
    """Display timing analysis report"""
    if 'timing_data' in st.session_state and st.session_state.timing_data:
        
        timing_data = st.session_state.timing_data
        total_time = sum(timing_data.values())
        
        # Create timing report
        timing_df = pd.DataFrame({
            'Process': list(timing_data.keys()),
            'Time (seconds)': list(timing_data.values()),
            'Percentage': [f"{(time/total_time)*100:.1f}%" for time in timing_data.values()]
        }).sort_values('Time (seconds)', ascending=False)
        
        
        slowest_time = timing_df.iloc[0]['Time (seconds)']
        st.metric("Затраченное время", f"{slowest_time:.2f} сек")

def main():
    st.set_page_config(
        page_title="Отчет Metaboscan",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Сайдбар с управлением Dash
    with st.sidebar:
        st.header("Управление Dash приложением")
        
        if st.button("🔄 Проверить статус Dash"):
            if check_dash_health():
                st.success("Dash приложение работает")
            else:
                st.error("Dash приложение не доступно")
        

    # Стили
    st.markdown("""
        <style>
            body { font-size: 14px !important; }
            .stTextInput input, .stNumberInput input, .stSelectbox select, .stDateInput input {
                font-size: 14px !important;
            }
            .stDataFrame { font-size: 14px !important; }
            .stButton button { font-size: 14px !important; }
        </style>
    """, unsafe_allow_html=True)

    # Инициализация состояния
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'pdf_data' not in st.session_state:
        st.session_state.pdf_data = None
    if 'pdf_filename' not in st.session_state:
        st.session_state.pdf_filename = None
    if 'dash_checked' not in st.session_state:
        if check_dash_health():
            st.sidebar.success("✓ Dash приложение работает")
        else:
            st.sidebar.warning("⚠️ Dash приложение не запущено")
        st.session_state.dash_checked = True
    if 'timing_data' not in st.session_state:
        st.session_state.timing_data = {}

    # Основная форма
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.form("report_form"):
            st.write("Информация о пациенте")
            
            cols = st.columns(4)
            with cols[0]:
                name = st.text_input("Полное имя (ФИО)", placeholder="Иванов Иван Иванович")
            with cols[1]:
                age = st.number_input("Возраст", min_value=0, max_value=120, value=47)
            with cols[2]:
                gender = st.selectbox("Пол", ("М", "Ж"), index=0)
            with cols[3]:
                date = st.date_input("Дата отчета", datetime.now(), format="DD.MM.YYYY")
            
            layout = st.selectbox("Тип отчета", ("basic", "recommendation"), index=0)
            
            st.write("Загрузите данные")
            metabolomic_data_file = st.file_uploader(
                "Метаболомный профиль пациента (Excel)",
                type=["xlsx", "xls"],
                key="metabolomic_data"
            )
            
            # Add checkbox for timing analysis
            enable_timing = st.checkbox("Включить анализ времени выполнения", value=True)
            
            submitted = st.form_submit_button("Сформировать отчет", type="primary")
    
    with col2:
        st.write("Редактирование параметров")
        
        REF_FILE = "Ref.xlsx"
        if os.path.exists(REF_FILE):
            try:
                if 'original_ref' not in st.session_state or 'edited_ref' not in st.session_state:
                    xls = pd.ExcelFile(REF_FILE)
                    st.session_state.original_ref = {
                        sheet_name: xls.parse(sheet_name) 
                        for sheet_name in xls.sheet_names
                    }
                    st.session_state.edited_ref = {
                        sheet_name: df.copy() 
                        for sheet_name, df in st.session_state.original_ref.items()
                    }
                
                tabs = st.tabs(list(st.session_state.edited_ref.keys()))
                
                for tab, (sheet_name, df) in zip(tabs, st.session_state.edited_ref.items()):
                    with tab:
                        edited_df = st.data_editor(
                            df,
                            column_config={
                            "веса": st.column_config.NumberColumn(
                                "веса",
                                min_value=0.0,
                                max_value=10.0,
                                step=0.01,
                            )},
                            num_rows="dynamic",
                            use_container_width=True,
                            height=300,
                            key=f"editor_{sheet_name}"
                        )
                        st.session_state.edited_ref[sheet_name] = edited_df
                
                if st.button("Сбросить изменения", key="reset_button"):
                    st.session_state.edited_ref = {
                        sheet_name: df.copy() 
                        for sheet_name, df in st.session_state.original_ref.items()
                    }
                    st.rerun()
                
            except Exception as e:
                st.error(f"Ошибка при загрузке файла параметров: {str(e)}")
        else:
            st.error(f"Файл параметров не найден: {REF_FILE}")
            st.session_state.edited_ref = None

    # Инициализация сообщений
    if 'doctor_message' not in st.session_state:
        st.session_state.doctor_message = ""
    if 'patient_message' not in st.session_state:
        st.session_state.patient_message = ""
    if 'patient_long_message' not in st.session_state:
        st.session_state.patient_long_message = ""
    
    # Обработка отправки формы
    if submitted:
        if validate_inputs(name, metabolomic_data_file):
            with st.spinner("🔬 Читаем данные и генерируем отчет..."):
                try:
                    # Read metabolomic data
                    metabolomic_data_df = read_metabolomic_data(metabolomic_data_file)
                    if metabolomic_data_df is None:
                        return
                    
                    # Prepare data dictionary with DataFrames
                    data_dict = {
                        "metabolomic_data": metabolomic_data_df,
                        "patient_info": {
                            "name": name.strip(),
                            "age": age,
                            "date": date.strftime("%d.%m.%Y"),
                            "gender": gender,
                            "layout": layout
                        }
                    }
                    
                    # Сохраняем результаты
                    st.session_state.processed_data = data_dict
                    st.session_state.pdf_data = None
                    st.session_state.pdf_filename = None
                    
                    if layout == "basic":
                        pdf_data, filename = generate_pdf_report_api(
                            st.session_state.processed_data["patient_info"],
                            st.session_state.processed_data
                        )
                        
                        if pdf_data:
                            st.session_state.pdf_data = pdf_data
                            st.session_state.pdf_filename = filename
                            st.success("✅ Отчет успешно сформирован!")
                    
                    elif layout == "recommendation":
                        st.info("Пожалуйста, заполните поля ниже для генерации отчета с рекомендациями")
            
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")

    # Кнопка скачивания PDF
    if st.session_state.pdf_data:
        st.download_button(
            label="📥 Скачать отчет",
            data=st.session_state.pdf_data,
            file_name=st.session_state.pdf_filename,
            mime="application/pdf",
            key="download_pdf"
        )
    
    # Форма рекомендаций
    if (st.session_state.processed_data and 
        st.session_state.processed_data["patient_info"]["layout"] == "recommendation"):
        
        with st.form("recommendation_form"):
            st.session_state.patient_message = st.text_area(
                value=st.session_state.patient_message,
                key="patient_message_input",
                height=150,
                max_chars=600,
                label="Текст для пациента",
                placeholder="Введите выводы для пациента..."
            )
            st.session_state.patient_long_message = st.text_area(
                value=st.session_state.patient_long_message,
                key="patient_long_message_input",
                height=150,
                max_chars=3000,
                label = "Текст для пациента (расширенный)",
                placeholder="Введите выводы для пациента (расширенный)..."
            )
            st.session_state.doctor_message = st.text_area(
                value=st.session_state.doctor_message,
                key="doctor_message_input",
                height=150,
                max_chars=3000,
                label="Текст для врача",
                placeholder="Введите выводы для врача..."
            )
            submitted_recommendation = st.form_submit_button("Сформировать отчет с рекомендациями", type="primary")
            
        if submitted_recommendation:
            with st.spinner("Генерируем отчет с рекомендациями..."):
                patient_info_with_messages = st.session_state.processed_data["patient_info"].copy()
                patient_info_with_messages["doctor_message"] = st.session_state.doctor_message
                patient_info_with_messages["patient_message"] = st.session_state.patient_message
                patient_info_with_messages["patient_long_message"] = st.session_state.patient_long_message
                
                st.session_state.pdf_data = None
                st.session_state.pdf_filename = None
                
                pdf_data, filename = generate_pdf_report_api(
                    patient_info_with_messages,
                    st.session_state.processed_data
                )
                
                if pdf_data:
                    st.session_state.pdf_data = pdf_data
                    st.session_state.pdf_filename = filename
                    st.success("✅ Отчет с рекомендациями успешно сформирован!")
                    st.download_button(
                        label="📥 Скачать отчет с рекомендациями",
                        data=st.session_state.pdf_data,
                        file_name=st.session_state.pdf_filename,
                        mime="application/pdf",
                        key="download_pdf_recomendation"
                    )
                else:
                    st.error("Ошибка при генерации отчета с рекомендациями")

    # Display timing report if enabled
    if enable_timing and st.session_state.timing_data:
        display_timing_report()

if __name__ == "__main__":
    main()
