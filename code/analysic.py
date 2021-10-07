#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd 
import numpy as np


# # 导入数据

# In[9]:


# 导入查看
tea = pd.read_csv('../data/tea.csv')
comment = pd.read_csv('../data/comment.csv')
display(tea.head(3))
display(comment.head(3))


# In[10]:


# 信息查看
display(tea.info())
display(comment.info())


# # 标题词云

# In[11]:


'''
标题由品牌与名称组成，值保留标题部分
'''
title = tea[['标题']]
def title_clean(s):
    try: # 剔除品牌
        s = s.split(']')[1]
    except:
        s = s.split(']')[0]
    finally: # 剔除数字
        s = s.split('（')[0]
    return s
title['标题'] = title['标题'].map(title_clean)
title


# In[43]:


'''
词云图
'''
import jieba
txt = ''.join(list(title['标题']))
word = jieba.lcut(txt)
text = ' '.join(word)

from IPython.display import Image 
import stylecloud


stylecloud.gen_stylecloud(
    text=text,
    collocations=False,
    font_path=r'‪C:\Windows\Fonts\msyh.ttc',
    icon_name='fas fa-mug-hot',
    
    output_name='../output/标题词云图.png'
)
 
Image(filename='../output/标题词云图.png')


# # 评分分布直方图

# In[15]:


'''
评分取值 0-10
分区间 (0,2] (2,4] (4,6] (6,8] (8,10]
'''
score = tea[['评分']]
score = pd.cut(score['评分'],bins=[0,2,4,6,8,10]).reset_index()
score = score['评分'].value_counts().reset_index()
score = score.sort_values(by='index')
score['index'] = score['index'].map(lambda x: str(x))
score


# In[16]:


'''
直方图
'''
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.faker import Faker
from pyecharts.globals import ThemeType

c = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.WONDERLAND))
    .add_xaxis(list(score['index']))
    .add_yaxis("数量", list(score['评分']), category_gap=0, color=Faker.rand_color())
    .set_global_opts(title_opts=opts.TitleOpts(title="评分分布"))
)

c.render('../output/评分分布.html')
c.render_notebook()


# # 品牌词云

# In[17]:


'''
计算各个品牌数
'''
brand = tea[['品牌']]
brand = brand['品牌'].value_counts().reset_index()
word = list(brand['index'])
counts = list(brand['品牌'])
data = list(zip(word,counts))
data


# In[18]:


'''
词云图
'''
import pyecharts.options as opts
from pyecharts.charts import WordCloud


c = (
    WordCloud(init_opts=opts.InitOpts(theme=ThemeType.WONDERLAND))
    .add(series_name="品牌词云", data_pair=data, word_size_range=[6, 66])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="品牌词云", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
        ),
        tooltip_opts=opts.TooltipOpts(is_show=True),
    )
)
c.render('../output/品牌词云.html')
c.render_notebook()


# # 产地热力地图

# In[19]:


'''
计算每个产地产了多少种茶 
'''
place = tea[['产地']]
place = place['产地'].value_counts().reset_index()
place


# In[20]:


'''
热力地图 
'''
from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.faker import Faker

pieces = [
    {"min": 0, "max": 80},
    {"min": 80, "max": 160},
    {"min": 160, "max": 240},
    {"min": 240, "max": 320},
    {"min": 320, "max": 400},
    {"min": 400}
]

c = (
    Map(init_opts=opts.InitOpts(theme=ThemeType.WONDERLAND))
    .add("", [list(z) for z in zip(list(place['index']), list(place['产地']))], "china")
    .set_global_opts(
        title_opts=opts.TitleOpts(title="产地地区分布"),
        visualmap_opts=opts.VisualMapOpts(max_=400,
                                          is_piecewise=True,
                                         pieces=pieces),
    )
)
c.render('../output/产地地区分布.html')
c.render_notebook()


# # 每一种茶类有多少品种

# In[21]:


'''
茶类字段有分一级，二级，以 > 隔开
计算每个一级品种有多少二级品种
'''
kind = tea[['茶类']]
kind['一级'] = kind['茶类'].map(lambda x: x.split('>')[0])
kind = kind['一级'].value_counts().reset_index()
kind


# In[22]:


'''
柱状图
'''
from pyecharts import options as opts
from pyecharts.charts import Bar


c = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.WONDERLAND))
    .add_xaxis(list(kind['index']))
    .add_yaxis("", list(kind['一级']))
    .set_global_opts(title_opts=opts.TitleOpts(title="茶类分布"))
)
c.render('../output/茶类分布.html')
c.render_notebook()


# # 热搜前10茶

# In[23]:


'''
清洗标题，并按热搜升序排序选出前20
'''
resou = tea[['标题','评分','品牌','产地','茶类','热搜排行']]
resou['标题'] = resou['标题'].map(title_clean)
resou = resou.sort_values(by='热搜排行')
resou = resou.iloc[:10,:]
resou


# In[24]:


'''
条形图 
'''
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts


table = Table()

headers = ['标题','评分','品牌','产地','茶类','热搜排行']
rows = np.array(resou)
rows = rows.tolist()
table.add(headers, rows)
table.set_global_opts(
    title_opts=ComponentTitleOpts(title="热搜前10")
)
table.render("../output/热搜前20.html")
table.render_notebook()


# # 评论时间走势图

# In[25]:


'''
计算维度为年月，每一年每一月的评论数走势
比较每一年的走势
14-20年
'''
comm_time = comment[['评论时间']]
comm_time['year'] = comm_time['评论时间'].map(lambda x: x.split('-')[0])
comm_time['month'] = comm_time['评论时间'].map(lambda x: x.split('-')[1])

_14 = comm_time[comm_time['year'] == '2014']['month'].value_counts().reset_index()
_14 = _14.sort_values(by='index')

_15 = comm_time[comm_time['year'] == '2015']['month'].value_counts().reset_index()
_15 = _15.sort_values(by='index')

_16 = comm_time[comm_time['year'] == '2016']['month'].value_counts().reset_index()
_16 = _16.sort_values(by='index')

_17 = comm_time[comm_time['year'] == '2017']['month'].value_counts().reset_index()
_17 = _17.sort_values(by='index')

_18 = comm_time[comm_time['year'] == '2018']['month'].value_counts().reset_index()
_18 = _18.sort_values(by='index')

_19 = comm_time[comm_time['year'] == '2019']['month'].value_counts().reset_index()
_19 = _19.sort_values(by='index')

_20 = comm_time[comm_time['year'] == '2020']['month'].value_counts().reset_index()
_20 = _20.sort_values(by='index')


# In[26]:


'''
折线图
'''
import pyecharts.options as opts
from pyecharts.charts import Line 

x_data = ['1','2','3','4','5','6','7','8','9','10','11','12']


c = (
    Line()
    .add_xaxis(xaxis_data=x_data)
    .add_yaxis(
        series_name="2014",
        #stack="总量",
        y_axis=list(_14['month']),
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        series_name="2015",
        #stack="总量",
        y_axis=list(_15['month']),
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        series_name="2016",
        #stack="总量",
        y_axis=list(_16['month']),
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        series_name="2017",
        #stack="总量",
        y_axis=list(_17['month']),
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        series_name="2018",
        #stack="总量",
        y_axis=list(_18['month']),
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        series_name="2019",
        #stack="总量",
        y_axis=list(_19['month']),
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        series_name="2020",
        #stack="总量",
        y_axis=list(_20['month']),
        label_opts=opts.LabelOpts(is_show=False),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="评论走势"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
    )
)
c.render('../output/每一年评论走势.html')
c.render_notebook()


# # 总评聚类

# In[27]:


'''
总评聚类，把相似的聚在一起
使用 textRank 算法提取每种茶总评的关键词
文本向量化，计算余弦相似度 
最后进行聚类
对每个聚类结果绘制词云
'''
zongping = tea[['总评']]


# textRank 算法
from jieba import analyse

def textrank_extract(text, keyword_num=10):
    textrank = analyse.textrank
    keywords = textrank(text, keyword_num)
    # 输出抽取出的关键词
    kw = ''
    for keyword in keywords:
        kw += keyword + ' '
    return kw

zongping['keyword'] = zongping['总评'].map(textrank_extract)
zongping


# In[28]:


# 文本向量化，计算余弦相似度
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vect = CountVectorizer()
X = vect.fit_transform(list(zongping['keyword']))  # 将关键词的内容文本向量化
X = X.toarray()

words_bag2 = vect.get_feature_names()  
df = pd.DataFrame(X, columns=words_bag2)
cosine_similarities  = cosine_similarity(df)

cosine_similarities


# In[29]:


# 聚类
from sklearn.cluster import KMeans
kms = KMeans(n_clusters=2, random_state=123)
k_data = kms.fit_predict(cosine_similarities)
zongping['flag'] = k_data
zongping


# In[42]:


# 绘制词云
from pyecharts.globals import SymbolType
from collections import Counter
pd.set_option('display.max_rows',None)

for i in range(2):
    t = '种类' + str(i+1)
    data = list(zongping[zongping['flag'] == i]['keyword'])
    data = list(dict(Counter(' '.join(data).split())).items())
    data = sorted(data,key=lambda x: x[1],reverse=True)[:250]
    c = (
        WordCloud(init_opts=opts.InitOpts(theme=ThemeType.ROMANTIC))
        .add(series_name=t, data_pair=data,
             word_size_range=[6, 66],shape=SymbolType.DIAMOND)
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=t, title_textstyle_opts=opts.TextStyleOpts(font_size=23)
            ),
            tooltip_opts=opts.TooltipOpts(is_show=True),
        )
    )
    display(c.render_notebook())


# # 评论关键词提取

# In[32]:


'''
分词，词性标注，去除停用词
'''
comm = comment[['评论内容']]

import jieba
import jieba.posseg as psg

# 把每条评论分词，并标注每个词的词性
worker = lambda s: [(x.word,x.flag) for x in psg.cut(str(s))] # 自定义简单分词函数
seg_word = comm['评论内容'].apply(worker)
seg_word.head()


# In[33]:


# 提取名词，形容词，删除停用词
stop_word_path = '../data/stopword.txt'
stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path,encoding='utf-8').readlines()]
comm = list(seg_word.values)
new_comm = []
for cm in comm:
    t = ''
    for c in cm:
        if (c[1] == 'n' or c[1] == 'adj') and (c[0] not in stopword_list):
            t += c[0] + ' '
    new_comm.append(t.strip())
new_comm


# ## TF_IDF 算法

# In[34]:


import math
import functools

# idf值统计方法
def train_idf(doc_list):
    '''
    训练数据集生成对应的 IDF 值字典
    :param doc_list:
    :return:
    '''
    idf_dic = {}
    # 总文档数，可以理解为几个文档，或者几条评论等
    tt_count = len(doc_list)

    # 每个词出现的文档数
    for doc in doc_list:
        for word in doc.split():
            idf_dic[word] = idf_dic.get(word, 0.0) + 1.0

    # 按公式转换为idf值，分母加1进行平滑处理
    for k, v in idf_dic.items():
        idf_dic[k] = math.log(tt_count / (1.0 + v))

    # 对于没有在字典中的词，默认其仅在一个文档出现，得到默认idf值
    default_idf = math.log(tt_count / (1.0))
    return idf_dic, default_idf


# TF-IDF类
class TfIdf(object):
    # 四个参数分别是：训练好的idf字典，默认idf值，处理后的待提取文本，关键词数量
    def __init__(self, idf_dic, default_idf, word_list, keyword_num):
        self.word_list = word_list
        self.idf_dic, self.default_idf = idf_dic, default_idf
        self.tf_dic = self.get_tf_dic()
        self.keyword_num = keyword_num

    # 统计tf值
    def get_tf_dic(self):
        tf_dic = {}
        for word in self.word_list:
            tf_dic[word] = tf_dic.get(word, 0.0) + 1.0

        tt_count = len(self.word_list)
        for k, v in tf_dic.items():
            tf_dic[k] = float(v) / tt_count

        return tf_dic

    # 按公式计算tf-idf
    def get_tfidf(self):
        tfidf_dic = {}
        for word in self.word_list:
            idf = self.idf_dic.get(word, self.default_idf)
            tf = self.tf_dic.get(word, 0)

            tfidf = tf * idf
            tfidf_dic[word] = tfidf

        tfidf_dic.items()
        # 根据tf-idf排序，去排名前keyword_num的词作为关键词
        for k, v in sorted(tfidf_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            print(k + "/ ", end='')
        print()


#  排序函数，用于topK关键词的按值排序，得分相同，再根据关键词排序
def cmp(e1, e2):
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1

        
# 计算 idf 值
idf_dic, default_idf = train_idf(new_comm)
# 调用计算好的 idf 值计算 TF-IDF 值，选出排名靠前的词语
word_list = []
for c in new_comm:
    c = c.split()
    for i in c:
        word_list.append(i)
tfidf_model = TfIdf(idf_dic, default_idf, word_list, keyword_num=10)
tfidf_model.get_tfidf()


# ## LDA 主题模型

# In[35]:


get_ipython().run_line_magic('matplotlib', 'inline')
import re
import itertools
import matplotlib
from gensim import corpora, models
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['SimHei'] # 解决中文乱码问题
plt.rcParams['axes.unicode_minus'] = False # 解决负号无法正常显示的问题
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg' # 将图标设置成矢量图格式显示，会更清晰")
plt.style.use('Solarize_Light2') # 设置绘图样式
#matplotlib.use('Qt5Agg')

# 建立词典,建立语料库
w_dict = corpora.Dictionary([[i] for i in word_list])
w_corpus = [w_dict.doc2bow(j) for j in [[i] for i in word_list]]  

# 寻找最优主题数
# 构造主题数寻优函数
def cos(vector1, vector2):  # 余弦相似度函数
    dot_product = 0.0;
    normA = 0.0;
    normB = 0.0;
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return (None)
    else:
        return (dot_product / ((normA * normB) ** 0.5))

    # 主题数寻优


def lda_k(x_corpus, x_dict):
    # 初始化平均余弦相似度
    mean_similarity = []
    mean_similarity.append(1)

    # 循环生成主题并计算主题间相似度
    for i in np.arange(2, 11):
        lda = models.LdaModel(x_corpus, num_topics=i, id2word=x_dict)  # LDA模型训练
        for j in np.arange(i):
            term = lda.show_topics(num_words=50)

        # 提取各主题词
        top_word = []
        for k in np.arange(i):
            top_word.append([''.join(re.findall('"(.*)"', i))                              for i in term[k][1].split('+')])  # 列出所有词

        # 构造词频向量
        word = sum(top_word, [])  # 列出所有的词
        unique_word = set(word)  # 去除重复的词

        # 构造主题词列表，行表示主题号，列表示各主题词
        mat = []
        for j in np.arange(i):
            top_w = top_word[j]
            mat.append(tuple([top_w.count(k) for k in unique_word]))

        p = list(itertools.permutations(list(np.arange(i)), 2))
        l = len(p)
        top_similarity = [0]
        for w in np.arange(l):
            vector1 = mat[p[w][0]]
            vector2 = mat[p[w][1]]
            top_similarity.append(cos(vector1, vector2))

        # 计算平均余弦相似度
        mean_similarity.append(sum(top_similarity) / l)
    return (mean_similarity)

# 计算主题平均余弦相似度
w_k = lda_k(w_corpus, w_dict)

# 绘制主题平均余弦相似度图形
font = FontProperties(size=14)

fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(211)
ax1.plot(w_k)
ax1.set_xlabel('LDA主题数寻优', fontproperties=font)

plt.show()


# In[39]:


# LDA主题分析
lda = models.LdaModel(w_corpus, num_topics = 1, id2word = w_dict)
topic = lda.print_topics(num_words = 10)

theme = []
for p in topic:
    word = re.findall('\*"(.*?)"',p[1])
    theme.append(word)
theme


# In[ ]:




