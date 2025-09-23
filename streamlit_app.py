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
        # Safe session state access
        if 'timing_data' not in st.session_state:
            st.session_state.timing_data = {}
        st.session_state.timing_data[func.__name__] = execution_time
        return result
    return wrapper

DASH_APP_URL = "https://metaboreport-test-rezvanov.amvera.io/"

def initialize_session_state():
    """Initialize all session state variables safely"""
    default_state = {
        'processed_data': None,
        'pdf_data': None,
        'pdf_filename': None,
        'dash_checked': False,
        'timing_data': {},
        'session_id': None,
        'doctor_message': "",
        'patient_message': "",
        'patient_long_message': "",
        'form_submitted': False
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

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
        response = requests.post(url, json=payload, timeout=90)

        if response.status_code == 200:
            response_data = response.json()
            session_id = response_data.get('session_id')
            if session_id:
                st.session_state.session_id = session_id
            return True, "Data updated successfully", session_id
        else:
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
    if not name or not name.strip():
        st.error("Please enter a valid patient name")
        return False
    if file1 is None:
        st.error("Please upload metabolomic data file")
        return False
    return True

@time_function
def download_pdf_from_dash(session_id):
    """Загрузка PDF из Dash приложения с использованием session_id"""
    try:
        response = requests.get(
            f"{DASH_APP_URL}/download_pdf/{session_id}",
            timeout=90,
        )

        if response.status_code == 200:
            pdf_data = response.content
            content_disposition = response.headers.get('Content-Disposition', '')
            filename = f"MetaboScan_Report.pdf"

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
        response = requests.get(f"{DASH_APP_URL}/health", timeout=30)
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

        if total_time > 0:
            timing_df = pd.DataFrame({
                'Process': list(timing_data.keys()),
                'Time (seconds)': list(timing_data.values()),
                'Percentage': [f"{(time/total_time)*100:.1f}%" for time in timing_data.values()]
            }).sort_values('Time (seconds)', ascending=False)

            if not timing_df.empty:
                slowest_time = timing_df.iloc[0]['Time (seconds)']
                st.metric("Затраченное время", f"{slowest_time:.2f} сек")

def main():
    # Инициализация многопользовательской системы
    user_key = initialize_user_session()
    user_data = get_user_data(user_key)
    
    # Очистка старых сессий при каждой загрузке
    cleanup_inactive_sessions()
    
    st.set_page_config(
        page_title="Отчет Metaboscan",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Отображаем информацию о текущих пользователях в сайдбаре
    with st.sidebar:
        st.header("Управление Dash приложением")
        st.info(f"Активных сессий: {len(st.session_state.get('multi_user_sessions', {}))}")
        
        if st.button("🔄 Проверить статус Dash"):
            if check_dash_health():
                st.success("Dash приложение работает")
                set_user_data(user_key, {'dash_checked': True})
            else:
                st.error("Dash приложение не доступно")
        
        # Информация о текущем пользователе
        if user_data.get('session_id'):
            st.info(f"Session ID: {user_data['session_id'][:8]}...")
        
        # Кнопка очистки текущей сессии
        if st.button("🧹 Очистить мою сессию"):
            set_user_data(user_key, {
                'processed_data': None,
                'pdf_data': None,
                'pdf_filename': None,
                'session_id': None
            })
            st.rerun()

    # Основная форма с использованием данных конкретного пользователя
    with st.form("report_form", clear_on_submit=False):
        st.write("Информация о пациенте")

        cols = st.columns(4)
        with cols[0]:
            name = st.text_input("Полное имя (ФИО)", 
                               placeholder="Иванов Иван Иванович",
                               key=f"name_{user_key}")  # Уникальный ключ
        with cols[1]:
            age = st.number_input("Возраст", min_value=0, max_value=120, 
                                value=47, key=f"age_{user_key}")
        with cols[2]:
            gender = st.selectbox("Пол", ("М", "Ж"), 
                                index=0, key=f"gender_{user_key}")
        with cols[3]:
            date = st.date_input("Дата отчета", datetime.now(), 
                               format="DD.MM.YYYY", key=f"date_{user_key}")

        layout = st.selectbox("Тип отчета", ("basic", "recommendation"), 
                            index=0, key=f"layout_{user_key}")

        st.write("Загрузите данные")
        metabolomic_data_file = st.file_uploader(
            "Метаболомный профиль пациента (Excel)",
            type=["xlsx", "xls"],
            key=f"metabolomic_data_{user_key}"  # Уникальный ключ для каждого пользователя
        )

        enable_timing = st.checkbox("Включить анализ времени выполнения", 
                                  value=True, key=f"timing_{user_key}")

        submitted = st.form_submit_button("Сформировать отчет", type="primary")

    # Обработка отправки формы для конкретного пользователя
    if submitted:
        if validate_inputs(name, metabolomic_data_file):
            with st.spinner("🔬 Читаем данные и генерируем отчет..."):
                try:
                    # Read metabolomic data
                    metabolomic_data_df = read_metabolomic_data(metabolomic_data_file)
                    if metabolomic_data_df is None:
                        return

                    # Prepare data for current user
                    data_dict = {
                        "metabolomic_data": metabolomic_data_df,
                    }

                    patient_info = {
                        "name": name.strip(),
                        "age": age,
                        "date": date.strftime("%d.%m.%Y"),
                        "gender": gender,
                        "layout": layout
                    }

                    # Сохраняем результаты для текущего пользователя
                    set_user_data(user_key, {
                        'processed_data': {
                            "patient_info": patient_info,
                            "data_dict": data_dict
                        },
                        'pdf_data': None,
                        'pdf_filename': None
                    })

                    if layout == "basic":
                        pdf_data, filename = generate_pdf_report_api(
                            patient_info,
                            data_dict
                        )

                        if pdf_data:
                            set_user_data(user_key, {
                                'pdf_data': pdf_data,
                                'pdf_filename': filename
                            })
                            st.success("✅ Отчет успешно сформирован!")

                    elif layout == "recommendation":
                        st.info("Пожалуйста, заполните поля ниже для генерации отчета с рекомендациями")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    # Кнопка скачивания PDF для текущего пользователя
    current_pdf_data = user_data.get('pdf_data')
    current_filename = user_data.get('pdf_filename')
    
    if current_pdf_data:
        st.download_button(
            label="📥 Скачать отчет",
            data=current_pdf_data,
            file_name=current_filename,
            mime="application/pdf",
            key=f"download_{user_key}"  # Уникальный ключ
        )

    # Форма рекомендаций для текущего пользователя
    processed_data = user_data.get('processed_data')
    if (processed_data and 
        processed_data["patient_info"]["layout"] == "recommendation"):

        with st.form("recommendation_form"):
            # Используем текущие значения из сессии пользователя
            current_patient_message = user_data.get('patient_message', '')
            current_patient_long_message = user_data.get('patient_long_message', '')
            current_doctor_message = user_data.get('doctor_message', '')
            
            patient_message = st.text_area(
                value=current_patient_message,
                key=f"patient_msg_{user_key}",
                height=150,
                max_chars=600,
                label="Текст для пациента",
                placeholder="Введите выводы для пациента..."
            )
            
            patient_long_message = st.text_area(
                value=current_patient_long_message,
                key=f"patient_long_msg_{user_key}",
                height=150,
                max_chars=3000,
                label="Текст для пациента (расширенный)",
                placeholder="Введите выводы для пациента (расширенный)..."
            )
            
            doctor_message = st.text_area(
                value=current_doctor_message,
                key=f"doctor_msg_{user_key}",
                height=150,
                max_chars=3000,
                label="Текст для врача",
                placeholder="Введите выводы для врача..."
            )
            
            submitted_recommendation = st.form_submit_button("Сформировать отчет с рекомендациями", type="primary")

        if submitted_recommendation:
            # Сохраняем сообщения для текущего пользователя
            set_user_data(user_key, {
                'patient_message': patient_message,
                'patient_long_message': patient_long_message,
                'doctor_message': doctor_message
            })
            
            with st.spinner("Генерируем отчет с рекомендациями..."):
                patient_info_with_messages = processed_data["patient_info"].copy()
                patient_info_with_messages["doctor_message"] = doctor_message
                patient_info_with_messages["patient_message"] = patient_message
                patient_info_with_messages["patient_long_message"] = patient_long_message

                pdf_data, filename = generate_pdf_report_api(
                    patient_info_with_messages,
                    processed_data["data_dict"]
                )

                if pdf_data:
                    set_user_data(user_key, {
                        'pdf_data': pdf_data,
                        'pdf_filename': filename
                    })
                    st.success("✅ Отчет с рекомендациями успешно сформирован!")
                else:
                    st.error("Ошибка при генерации отчета с рекомендациями")

    # Display timing report for current user
    if enable_timing and user_data.get('timing_data'):
        display_timing_report(user_data['timing_data'])

def display_timing_report(timing_data):
    """Display timing analysis report for specific user"""
    if timing_data:
        total_time = sum(timing_data.values())
        if total_time > 0:
            timing_df = pd.DataFrame({
                'Process': list(timing_data.keys()),
                'Time (seconds)': list(timing_data.values()),
                'Percentage': [f"{(time/total_time)*100:.1f}%" for time in timing_data.values()]
            }).sort_values('Time (seconds)', ascending=False)

            if not timing_df.empty:
                slowest_time = timing_df.iloc[0]['Time (seconds)']
                st.metric("Затраченное время", f"{slowest_time:.2f} сек")

if __name__ == "__main__":
    main()
