# Student-essays
The first block of student essay analysis.

Document under analysis: 5th-year student essay in Chinese, written as part of the "Language for Professional Communication" course

Stages of analysis:

1. text preparation (punctuation removal, spaces, tokenization)
2. character counting, token counting.
3. plotting a Zipf graph
4. determining via a Zipf graph whether the text was written by a student or a neural network.
5. constructing a word cloud
6. determining the TTR of the text
7. keyword analysis
8. constructing a keyword cloud (first 30)
9. determining the number of sentences and the average sentence length.
10. (checking using language_tool_python)
11. determining the presence of words from the word list in the textbook on the topic of the essay

import re
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import language_tool_python
from collections import Counter
from matplotlib import font_manager
from nltk import FreqDist
import matplotlib as mpl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy import displacy #изуализации синтаксических деревьев (зависимостей между словами), визуализациz распознавания именованных сущностей в тексте
# import jieba - тоже может токенизировать китайские тексты, но я не пробовала
from wordcloud import WordCloud

nlp = spacy.load('C:/Program Files/Python312/Lib/site-packages/zh_core_web_sm/zh_core_web_sm-3.8.0')
# nlp = spacy.load('zh_core_web_sm') #вообще-то нужно вот так, но у меня работает только усли указать путь вручную

# Тестируем модель на примере текста
doc = nlp("你好，世界！") 

# Выводим токены и их части речи
for token in doc:
    print(token.text, token.pos_)

# Путь к файлу с текстом
file_path = 'C:/Users/Nadezhda Ivanchenko/Documents/ДПО 2024-2025/проект 1/Студенческие сочинения/Karina.txt'

# Читаем текст из файла с кодировкой UTF-8
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Очищаем текст от пробелов, переводов строки и знаков препинания (делается один раз)
clean_text = re.sub(r'[\s, ，。！？、；：“”‘’;（）—…]', '', text)
# Подсчитываем количество иероглифов в очищённом тексте
char_count = len(clean_text)

# Загружаем модель spaCy для китайского языка
#nlp = spacy.load('zh_core_web_sm')
doc = nlp(text)

# Токенизация: получаем все токены (слова)
tokens = [token.text for token in doc]

# Токенизация: исключаем знаки препинания
tokens_without_punct = [token.text for token in doc if not token.is_punct and token.text.strip() != ""]

# Токенизация: исключаем стоп-слова
tokens_without_stopwords = [token.text for token in doc if not token.is_stop and token.text.strip() != ""]

# Подсчет количества токенов
count_tokens_without_punct = len(tokens_without_punct)
count_tokens_without_stopwords = len(tokens_without_stopwords)

# Выводим результаты
print(f"Количество иероглифов (без пробелов и знаков препинания): {char_count}")
print(f"Токены текста без знаков препинания ({count_tokens_without_punct} токенов):")
print(", ".join(tokens_without_punct))
print(f"\nТокены текста без стоп-слов ({count_tokens_without_stopwords} токенов):")
print(", ".join(tokens_without_stopwords))

# проверим, как выглядит текст без знаков препинания и пробелов
print(clean_text)

# График Ципфа

font_path = 'C:/Windows/Fonts/msyh.ttc'  # Путь к шрифту Microsoft YaHei
font_manager.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = 'Microsoft YaHei'  # Устанавливаем шрифт по умолчанию

text = clean_text

# Обрабатываем текст с помощью spaCy
doc = nlp(clean_text)

# --- Токенизация ---
# Токены без знаков препинания
tokens_without_punct = [token.text for token in doc if not token.is_punct and token.text.strip() != ""]

# Подсчет частот
word_freq_without_punct = Counter(tokens_without_punct)

# Создаем объекты FreqDist (из NLTK) для построения графиков
fdist_without_punct = FreqDist(word_freq_without_punct)

# График Ципфа для текста без знаков препинания
plt.figure(figsize=(12, 6))
fdist_without_punct.plot(20, cumulative=False)
plt.title('График Ципфа: текст без знаков препинания')
plt.xlabel('Слова')
plt.ylabel('Частота слова')
plt.show()

# кем написан текст?

word_freq = Counter(tokens_without_punct)

# Сортируем слова по убыванию частоты
sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

# Создаем список рангов (1, 2, 3, ..., N)
ranks = range(1, len(sorted_freq) + 1)
# Извлекаем частоты в порядке ранжирования
frequencies = [freq for word, freq in sorted_freq]

# Построение графика Ципфа
plt.figure(figsize=(12, 6))
plt.plot(ranks, frequencies, marker='o')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Ранг слова')
plt.ylabel('Частота слова')
plt.title('График Ципфа для текста')

# Рассчитываем коэффициент наклона (логарифмическая регрессия)
log_ranks = np.log(ranks)
log_frequencies = np.log(frequencies)
slope, intercept = np.polyfit(log_ranks, log_frequencies, 1)

# Простейшая эвристика: если наклон меньше -1.1, возможно текст сгенерирован нейросетью
if slope < -1.1:
    print("Скорее всего, текст сгенерирован нейросетью.")
else:
    print("Скорее всего, текст написан человеком.")

plt.show()

# Строим облако слов для текста без знаков препинания
wordcloud_tokens_without_punct = WordCloud(font_path=font_path, width=800, height=400, background_color="white").generate_from_frequencies(word_freq_without_punct)

# Визуализируем облака слов
plt.figure(figsize=(12, 6))

# Облако слов для текста без знаков препинания
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_tokens_without_punct, interpolation="bilinear")
plt.axis("off")
plt.title("Облако слов: без знаков препинания")


plt.tight_layout()
plt.show()

#Считаем TTR
ttr = len(set(tokens_without_stopwords)) / len(tokens_without_stopwords)
print(f"TTR: {ttr:.4f}")

#Ключевые слова
# Для TF-IDF преобразуем список токенов в строку, разделяя слова пробелами
text_for_tfidf = " ".join(tokens_without_stopwords)

# Инициализируем TfidfVectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([text_for_tfidf])
feature_names = vectorizer.get_feature_names_out()

# Получаем TF-IDF оценки для каждого слова в документе (так как документ один, берем первую строку)
tfidf_scores = tfidf_matrix.toarray()[0]

# Создаем словарь: слово -> TF-IDF оценка
word_tfidf = dict(zip(feature_names, tfidf_scores))

# Сортируем слова по убыванию TF-IDF оценки
sorted_word_tfidf = sorted(word_tfidf.items(), key=lambda x: x[1], reverse=True)

# Выводим топ-10 ключевых слов
print("Топ-10 ключевых слов:")
for word, score in sorted_word_tfidf[:10]:
    print(f"{word}: {score:.4f}")

# облако ключевых слов

text_for_tfidf = " ".join(tokens_without_stopwords)

# Инициализируем TfidfVectorizer и вычисляем TF-IDF для текста
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([text_for_tfidf])
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray()[0]

# Создаем словарь: слово -> TF-IDF оценка
word_tfidf = dict(zip(feature_names, tfidf_scores))

# Сортируем слова по убыванию TF-IDF оценки и выбираем топ-50
sorted_words = sorted(word_tfidf.items(), key=lambda x: x[1], reverse=True)
top_50 = dict(sorted_words[:50])

# Строим облако ключевых слов для топ-50 слов
wordcloud = WordCloud(
    font_path='C:/Windows/Fonts/msyh.ttc',  # используем шрифт Microsoft YaHei для китайских символов
    width=800,
    height=400,
    background_color='white'
).generate_from_frequencies(top_50)

# Отображаем облако слов
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Облако ключевых слов (Топ-30)')
plt.show()

# Средняя длинна предложения

with open(file_path, 'r', encoding='utf-8') as file:
    original_text = file.read()

# Разбиваем текст на предложения по китайскому знаку конца предложения "。"
sentences = [s.strip() for s in original_text.split("。") if s.strip() != ""]

# Считаем количество предложений
num_sentences = len(sentences)

# Загружаем модель spaCy для китайского языка
nlp = spacy.load('C:/Program Files/Python312/Lib/site-packages/zh_core_web_sm/zh_core_web_sm-3.8.0')

# Токенизируем каждое предложение и суммируем количество токенов
total_tokens = 0
for s in sentences:
    doc_sent = nlp(s)
    tokens = [token.text for token in doc_sent if not token.is_space]
    total_tokens += len(tokens)

# Вычисляем среднюю длину предложения (если предложения есть)
avg_sentence_length = total_tokens / num_sentences if num_sentences > 0 else 0

# Выводим результаты
print(f"Общее количество предложений: {num_sentences}")
print(f"Общее количество слов: {total_tokens}")
print(f"Средняя длина предложения: {avg_sentence_length:.2f} слов")

# Грамматические ошибки (проверка с использованием language_tool_python)

# Создаем объект для проверки грамматических ошибок для китайского языка
tool = language_tool_python.LanguageTool('zh')

# Проверяем оригинальный текст на ошибки
matches = tool.check(original_text)

# Выводим общее количество найденных ошибок
print(f"Найдено ошибок: {len(matches)}\n")

# Выводим подробную информацию для первых 10 ошибок (если их больше 10)
for i, match in enumerate(matches[:10], 1):
    print(f"Ошибка {i}:")
    print(f"  Сообщение: {match.message}")
    print(f"  Контекст: {match.context}")
    print(f"  Неправильный фрагмент: {original_text[match.offset:match.offset+match.errorLength]}")
    if match.replacements:
        print(f"  Возможные исправления: {', '.join(match.replacements)}")
    else:
        print("  Возможных исправлений не предложено.")
    print("-" * 60)

# ищем в текстах сочинений (исходный текст) есть ли там слова из темы урока

target_tokens = ['湖畔', '成千上万', '湛蓝', '观赏', '语境', '顺理成章', '汇聚', '表象', '所在', '自然观', '历史感', '左右', '走向', '基于', '考量', '驱动', '有意无意', '遮蔽', '自在的', '年深日久', '窘境', '若是', '改编', '美饰', '源头', '因应', '自发进行', '源头', '博取', '欢心', '实实在在', '青藏高原', '藏族', '藏族人', '藏民族文化', '游牧', '农耕', '全球化', '多元文化', '弱势文化', '弱势','文化保护', '文化灭绝', '灭绝']

tokens = [token.text for token in doc]

# Создаем словарь только с найденными токенами и их частотой
found_tokens = {token: tokens.count(token) for token in target_tokens if tokens.count(token) > 0}

# Выводим найденные токены
if found_tokens:
    for token, count in found_tokens.items():
        print(f"Токен '{token}' найден {count} раз(а).")
else:
    print("Ни один из искомых токенов не найден в тексте.")
