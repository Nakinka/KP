import streamlit as st
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import datetime
import warnings
warnings.filterwarnings('ignore')

# Настройки страницы
st.set_page_config(
    page_title="Классификатор инцидентов",
    layout="wide",
    page_icon="🔧"
)

st.markdown("""
<style>
    div.stButton > button:first-child {
        background-color: #28a745;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #218838;
        color: white;
        border: none;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Заголовок приложения
st.title("🔧 Классификатор IT-инцидентов")
st.markdown("Определение категории и срочности инцидента на основе текстового описания")

# Функция для загрузки моделей с обработкой памяти
@st.cache_resource(show_spinner=False)
def load_models(model_dir):
    """Загружает модели и кодировщики с оптимизацией памяти"""
    try:
        # Очистка памяти перед загрузкой
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        st.info(f"🔄 Загрузка модели из: {model_dir}")
        
        # Проверяем существование папки
        if not os.path.exists(model_dir):
            st.error(f"❌ Папка {model_dir} не существует")
            return None, None, None, None, None
        
        # Загружаем токенизатор (из основной папки)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Загружаем модели из отдельных папок с оптимизацией памяти
        category_model_path = f"{model_dir}/category_model"
        urgency_model_path = f"{model_dir}/urgency_model"
        
        if not os.path.exists(category_model_path):
            st.error(f"❌ Папка категорий не найдена: {category_model_path}")
            return None, None, None, None, None
            
        if not os.path.exists(urgency_model_path):
            st.error(f"❌ Папка срочности не найдена: {urgency_model_path}")
            return None, None, None, None, None
        
        # Загружаем модели с оптимизацией памяти
        st.info("📥 Загрузка модели категорий...")
        category_model = AutoModelForSequenceClassification.from_pretrained(
            category_model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        st.info("📥 Загрузка модели срочности...")
        urgency_model = AutoModelForSequenceClassification.from_pretrained(
            urgency_model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Переводим в режим оценки и отключаем градиенты
        category_model.eval()
        urgency_model.eval()
        
        # Загружаем кодировщики
        le_category_path = f"{model_dir}/le_category.pkl"
        le_urgency_path = f"{model_dir}/le_urgency.pkl"
        
        if os.path.exists(le_category_path) and os.path.exists(le_urgency_path):
            le_category = pd.read_pickle(le_category_path)
            le_urgency = pd.read_pickle(le_urgency_path)
        else:
            st.error(f"❌ Не найдены файлы кодировщиков")
            return None, None, None, None, None
        
        st.success("✅ Модели успешно загружены!")
        return tokenizer, category_model, urgency_model, le_category, le_urgency
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки моделей: {str(e)}")
        return None, None, None, None, None

# Функция для предсказания
def predict_incident(text, tokenizer, category_model, urgency_model, le_category, le_urgency, max_len=100):
    """Предсказывает категорию и срочность инцидента"""
    try:
        # Токенизация текста
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        # Предсказание категории
        with torch.no_grad():
            category_outputs = category_model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"]
            )
            category_probs = torch.nn.functional.softmax(category_outputs.logits, dim=-1)
            category_pred = torch.argmax(category_probs, dim=1).item()
            category_confidence = category_probs[0][category_pred].item()
        
        # Предсказание срочности
        with torch.no_grad():
            urgency_outputs = urgency_model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"]
            )
            urgency_probs = torch.nn.functional.softmax(urgency_outputs.logits, dim=-1)
            urgency_pred = torch.argmax(urgency_probs, dim=1).item()
            urgency_confidence = urgency_probs[0][urgency_pred].item()
        
        # Декодируем предсказания
        category_name = le_category.inverse_transform([category_pred])[0]
        urgency_name = le_urgency.inverse_transform([urgency_pred])[0]
        
        # Вероятности для всех классов
        category_all_probs = {
            le_category.inverse_transform([i])[0]: category_probs[0][i].item() 
            for i in range(len(le_category.classes_))
        }
        
        urgency_all_probs = {
            le_urgency.inverse_transform([i])[0]: urgency_probs[0][i].item() 
            for i in range(len(le_urgency.classes_))
        }
        
        return {
            'category': category_name,
            'urgency': urgency_name,
            'category_confidence': category_confidence,
            'urgency_confidence': urgency_confidence,
            'category_all_probs': category_all_probs,
            'urgency_all_probs': urgency_all_probs
        }
    except Exception as e:
        st.error(f"❌ Ошибка при предсказании: {str(e)}")
        return None

# Функция для определения цвета delta
def get_delta_color(confidence, threshold_high=0.8, threshold_medium=0.6):
    """Определяет цвет для delta в st.metric"""
    if confidence >= threshold_high:
        return "normal"  # Зеленый - высокая уверенность
    elif confidence >= threshold_medium:
        return "off"     # Серый - средняя уверенность
    else:
        return "inverse" # Красный - низкая уверенность

def main():
    # Информация о памяти
    st.sidebar.title("⚙️ Настройки")
    st.sidebar.info("💡 Для работы приложения рекомендуется увеличить файл подкачки Windows")
    
    # Автоматический поиск последней модели
    models_dir = "models"
    available_models = []
    
    if os.path.exists(models_dir):
        available_models = [d for d in os.listdir(models_dir) 
                          if d.startswith("dual_rubert_") and os.path.isdir(os.path.join(models_dir, d))]
        available_models.sort(reverse=True)
    
    if available_models:
        selected_model = st.sidebar.selectbox(
            "Выберите модель:",
            available_models,
            index=0
        )
        model_path = os.path.join(models_dir, selected_model)
        st.sidebar.success(f"📁 Модель: {selected_model}")
        
    else:
        st.sidebar.error("❌ Модели не найдены в папке 'models'")
        st.stop()
    
    # Загрузка моделей с прогресс-баром
    with st.spinner("🔄 Загрузка моделей... Это может занять несколько минут"):
        tokenizer, category_model, urgency_model, le_category, le_urgency = load_models(model_path)
    
    if tokenizer is None or category_model is None or urgency_model is None:
        st.error("""
        ❌ Не удалось загрузить модели. Возможные причины:
        
        1. **Недостаточно памяти** - увеличьте файл подкачки Windows
        2. **Поврежденные файлы моделей** - проверьте целостность файлов
        3. **Нехватка оперативной памяти** - закройте другие приложения
        
        💡 **Решение:** Увеличьте файл подкачки до 8 ГБ через:
        - Панель управления → Система → Дополнительные параметры системы
        - Быстродействие → Параметры → Дополнительно → Виртуальная память
        """)
        return
    
    # Информация о модели в сайдбаре
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Информация о модели")
    
    # Основная область ввода
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Описание инцидента")
        
        # Варианты быстрого ввода
        example_incidents = {
            "Сервер не отвечает, все системы недоступны": "Аппаратный сбой, Высокий",
            "Приложение вылетает при открытии отчета": "Сбой приложения, Средний",
            "Медленная работа почтового клиента": "Ухудшение сервиса, Высокий",
            "Обнаружена попытка несанкционированного доступа": "Нарушение безопасности, Высокий",
            "Пропал интернет в филиале": "Потеря связи, Средний"
        }
        
        selected_example = st.selectbox(
            "Или выберите пример:",
            [""] + list(example_incidents.keys())
        )
        
        if selected_example:
            incident_text = st.text_area(
                "Опишите инцидент:",
                value=selected_example,
                height=150,
                placeholder="Опишите проблему подробно..."
            )
            st.caption(f"🎯 Ожидаемый результат: {example_incidents[selected_example]}")
        else:
            incident_text = st.text_area(
                "Опишите инцидент:",
                height=150,
                placeholder="Например: 'Сервер не отвечает, пользователи не могут работать...'"
            )
    
    with col2:
        st.subheader("📈 Статистика")
        
        # Информация о категориях
        st.markdown("**📂 Доступные категории:**")
        for i, category in enumerate(le_category.classes_):
            st.markdown(f"• {category}")
        
        st.markdown("**⚡ Уровни срочности:**")
        for urgency in le_urgency.classes_:
            st.markdown(f"• {urgency}")
    
    # Кнопка предсказания
    if st.button("🎯 Классифицировать инцидент", type="primary", use_container_width=True):
        if incident_text.strip():
            with st.spinner("🔍 Анализируем инцидент..."):
                result = predict_incident(
                    incident_text, 
                    tokenizer, 
                    category_model, 
                    urgency_model, 
                    le_category, 
                    le_urgency
                )
            
            if result is None:
                st.error("❌ Произошла ошибка при классификации инцидента")
                return
            
            # Отображаем результаты
            st.markdown("---")
            st.subheader("📊 Результаты классификации")
            
            tab1, tab2 = st.tabs(["📋 Результаты", "📊 Визуализация"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Категория
                    category_delta_color = get_delta_color(result['category_confidence'])
                    st.metric(
                        label="📂 Категория",
                        value=result['category'],
                        delta=f"Уверенность: {result['category_confidence']:.1%}",
                        delta_color=category_delta_color
                    )
                    
                    # Таблица вероятностей категорий
                    st.markdown("**📊 Вероятности по категориям:**")
                    category_probs_df = pd.DataFrame(
                        list(result['category_all_probs'].items()),
                        columns=['Категория', 'Вероятность']
                    ).sort_values('Вероятность', ascending=False)
                    
                    st.dataframe(
                        category_probs_df.style.format({'Вероятность': '{:.2%}'}),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    # Срочность
                    urgency_delta_color = get_delta_color(result['urgency_confidence'])
                    
                    st.metric(
                        label="⚡ Срочность",
                        value=result['urgency'],
                        delta=f"Уверенность: {result['urgency_confidence']:.1%}",
                        delta_color=urgency_delta_color
                    )
                    
                    # Таблица вероятностей срочности
                    st.markdown("**📊 Вероятности по срочности:**")
                    urgency_probs_df = pd.DataFrame(
                        list(result['urgency_all_probs'].items()),
                        columns=['Срочность', 'Вероятность']
                    ).sort_values('Вероятность', ascending=False)
                    
                    st.dataframe(
                        urgency_probs_df.style.format({'Вероятность': '{:.2%}'}),
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Индикаторы срочности
                st.markdown("---")
                st.subheader("🚨 Индикаторы срочности")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    urgency_icons = {"Высокий": "🔴", "Средний": "🟡", "Низкий": "🟢"}
                    icon = urgency_icons.get(result['urgency'], "⚪")
                    st.markdown(f"### {icon} {result['urgency']}")
                
                with col2:
                    confidence_level = "Высокая" if result['urgency_confidence'] > 0.7 else "Средняя" if result['urgency_confidence'] > 0.5 else "Низкая"
                    st.markdown(f"**🎯 Уверенность:** {confidence_level}")
                
                with col3:
                    if result['urgency'] == "Высокий":
                        st.error("🚨 Требуется немедленное вмешательство")
                    elif result['urgency'] == "Средний":
                        st.warning("⚠️ Требует внимания в рабочее время")
                    else:
                        st.success("✅ Может быть отложено")
                
                # Рекомендации по срочности
                st.markdown("---")
                st.subheader("💡 Рекомендации")
                
                urgency_recommendations = {
                    "Высокий": "🚨 Немедленные действия требуются! Немедленно уведомить ответственную команду, начать устранение в приоритетном порядке.",
                    "Средний": "⚠️ Требует внимания в рабочее время. Уведомить команду, планировать устранение в ближайшее время.",
                    "Низкий": "✅ Может быть отложено. Запланировать устранение в обычном порядке, мониторить ситуацию."
                }
                
                recommendation = urgency_recommendations.get(result['urgency'], "📋 Оцените ситуацию самостоятельно.")
                st.info(recommendation)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Распределение вероятностей по категориям**")
                    st.bar_chart(category_probs_df.set_index('Категория')['Вероятность'])
                
                with col2:
                    st.markdown("**📈 Распределение вероятностей по срочности**")
                    st.bar_chart(urgency_probs_df.set_index('Срочность')['Вероятность'])
            
        else:
            st.warning("⚠️ Пожалуйста, введите описание инцидента")

    # Раздел для пакетной обработки
    st.markdown("---")
    st.subheader("📦 Пакетная обработка")
    
    st.info("💡 Для пакетной обработки загрузите CSV файл с колонкой 'text'")
    
    uploaded_file = st.file_uploader("Загрузите CSV файл с инцидентами", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write(f"📊 Загружено {len(batch_data)} записей")
            
            if 'text' in batch_data.columns:
                if st.button("🔄 Обработать все записи", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []
                    
                    for i, row in enumerate(batch_data.itertuples()):
                        status_text.text(f"Обработка {i+1}/{len(batch_data)} записей...")
                        result = predict_incident(
                            str(row.text), 
                            tokenizer, 
                            category_model, 
                            urgency_model, 
                            le_category, 
                            le_urgency
                        )
                        if result is not None:
                            results.append({
                                'original_text': row.text,
                                'category': result['category'],
                                'urgency': result['urgency'],
                                'category_confidence': result['category_confidence'],
                                'urgency_confidence': result['urgency_confidence']
                            })
                        progress_bar.progress((i + 1) / len(batch_data))
                    
                    status_text.text("✅ Обработка завершена!")
                    
                    if results:
                        # Создаем DataFrame с результатами
                        results_df = pd.DataFrame(results)
                        
                        # Показываем результаты
                        st.subheader("📋 Результаты пакетной обработки")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Скачивание результатов
                        csv = results_df.to_csv(index=False, encoding='utf-8')
                        st.download_button(
                            label="💾 Скачать результаты (CSV)",
                            data=csv,
                            file_name=f"incident_classification_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.error("❌ Не удалось обработать ни одной записи")
            else:
                st.error("❌ CSV файл должен содержать колонку 'text' с описаниями инцидентов")
                
        except Exception as e:
            st.error(f"❌ Ошибка при обработке файла: {e}")

# Запуск приложения
if __name__ == "__main__":
    main()