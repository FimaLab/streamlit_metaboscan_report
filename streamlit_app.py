import os
import streamlit as st
import pandas as pd
from datetime import datetime
import time
import requests
import numpy as np
from functools import wraps

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

# Определяем URL Dash приложения в зависимости от окружения
if os.environ.get('DOCKER_ENV'):
    DASH_APP_URL = "http://dash:8050"  # для Docker контейнера
else:
    DASH_APP_URL = "http://localhost:80"  # для локальной разработки


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
def generate_pdf_report_api(patient_info, data_dict):
    """Генерация PDF отчета через Dash API"""
    try:
        # Prepare the data payload
        payload = {
            'name': patient_info['name'],
            'age': patient_info['age'],
            'date': patient_info['date'],
            'gender': patient_info['gender'],
            'layout': patient_info['layout'],
            'metabolomic_data': dataframe_to_dict(data_dict['metabolomic_data'])
        }
        
        # Добавление опциональных сообщений только если они есть
        for key in ['doctor_message', 'patient_message', 'patient_long_message']:
            if key in patient_info and patient_info[key]:
                payload[key] = patient_info[key]
        
        # Прогресс бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Подготовка данных...")
        progress_bar.progress(10)
        
        status_text.text("Отправка запроса на сервер...")
        progress_bar.progress(30)
        
        response = requests.post(
            f"{DASH_APP_URL}/generate_report",
            json=payload,
            timeout=90,
        )
        
        progress_bar.progress(70)
        status_text.text("Обработка ответа...")
        
        if response.status_code == 200:
            pdf_data = response.content
            
            # ГЕНЕРИРУЕМ ИМЯ ФАЙЛА НА ОСНОВЕ ДАННЫХ ПАЦИЕНТА
            filename = generate_filename(patient_info)
            
            progress_bar.progress(100)
            status_text.text("Готово!")
            status_text.empty()
            progress_bar.empty()
            
            return pdf_data, filename
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get('message', f'HTTP Error {response.status_code}')
            except (ValueError, TypeError):
                error_msg = response.text.strip() or f'HTTP Error {response.status_code}'
            
            progress_bar.empty()
            status_text.text(f"Ошибка: {error_msg}")
            status_text.empty()
            return None, None
                
    except requests.exceptions.Timeout:
        st.error("Сервер не ответил за отведенное время. Попробуйте уменьшить объем данных или подождите.")
        return None, None
    except Exception as e:
        st.error(f"Ошибка при отправке запроса: {str(e)}")
        return None, None

@time_function
def generate_filename(patient_info):
    """Генерирует имя файла на основе данных пациента"""
    name = patient_info['name'].replace(' ', '_')
    layout = patient_info['layout']
    date = patient_info['date'].replace('.', '-')
    
    if layout == "basic":
        return f"Metaboscan_{name}.pdf"
    elif layout == "recommendation":
        return f"Metaboscan+_{name}.pdf"
    else:
        return f"Metaboscan_{name}.pdf"
    
@time_function
def check_dash_health():
    """Проверка доступности Dash приложения"""
    try:
        response = requests.get(f"{DASH_APP_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

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
    if 'doctor_message' not in st.session_state:
        st.session_state.doctor_message = ""
    if 'patient_message' not in st.session_state:
        st.session_state.patient_message = ""
    if 'patient_long_message' not in st.session_state:
        st.session_state.patient_long_message = ""
    if 'current_layout' not in st.session_state:
        st.session_state.current_layout = "basic"

    # Основная форма
    with st.form("report_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Информация о пациенте")
            
            name = st.text_input("Полное имя (ФИО)", placeholder="Иванов Иван Иванович")
            
            cols = st.columns(3)
            with cols[0]:
                age = st.number_input("Возраст", min_value=0, max_value=120, value=47)
            with cols[1]:
                gender = st.selectbox("Пол", ("М", "Ж"), index=0)
            with cols[2]:
                date = st.date_input("Дата отчета", datetime.now(), format="DD.MM.YYYY")
            
            layout = st.selectbox("Тип отчета", ("basic", "recommendation"), index=0)
            
            st.write("### Загрузите данные")
            metabolomic_data_file = st.file_uploader(
                "Метаболомный профиль пациента (Excel)",
                type=["xlsx", "xls"],
                key="metabolomic_data"
            )
            
            enable_timing = st.checkbox("Включить анализ времени выполнения", value=True)
            
            submitted = st.form_submit_button("Сформировать отчет", type="primary")
        
        with col2:
            st.write("### Рекомендации")
            
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
                label="Текст для пациента (расширенный)",
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

    # Обработка отправки формы
    if submitted:
        if validate_inputs(name, metabolomic_data_file):
            with st.spinner("🔬 Читаем данные и генерируем отчет..."):
                try:
                    # Read metabolomic data
                    metabolomic_data_df = read_metabolomic_data(metabolomic_data_file)
                    if metabolomic_data_df is None:
                        return
                    
                    # Prepare patient info
                    patient_info = {
                        "name": name.strip(),
                        "age": age,
                        "date": date.strftime("%d.%m.%Y"),
                        "gender": gender,
                        "layout": layout
                    }
                    
                    # Add messages only for recommendation layout
                    if layout == "recommendation":
                        patient_info["doctor_message"] = st.session_state.doctor_message
                        patient_info["patient_message"] = st.session_state.patient_message
                        patient_info["patient_long_message"] = st.session_state.patient_long_message
                    
                    # Prepare data dictionary
                    data_dict = {
                        "metabolomic_data": metabolomic_data_df
                    }
                    
                    # Генерируем PDF (теперь возвращает и данные и имя файла)
                    pdf_data, filename = generate_pdf_report_api(patient_info, data_dict)
                    
                    if pdf_data:
                        st.session_state.pdf_data = pdf_data
                        st.session_state.pdf_filename = filename  # Сохраняем сгенерированное имя файла
                        st.session_state.current_layout = layout
                        st.success("✅ Отчет успешно сформирован!")
                        st.rerun()
                    else:
                        st.error("Не удалось сгенерировать отчет")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")

    # Кнопка скачивания PDF
    if st.session_state.pdf_data:
        st.markdown("---")
        st.write("### Скачать отчет")
        
        st.download_button(
            label="📥 Скачать PDF отчет",
            data=st.session_state.pdf_data,
            file_name=st.session_state.pdf_filename,
            mime="application/pdf",
            key="download_pdf"
        )
        

    # Display timing report if enabled
    if enable_timing and st.session_state.timing_data:
        st.markdown("---")
        display_timing_report()

if __name__ == "__main__":
    main()
