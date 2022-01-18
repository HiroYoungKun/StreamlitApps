import streamlit as st
from PIL import Image
import numpy as np
import docx
import pandas as pd    
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *
from wordcloud import WordCloud
from matplotlib import font_manager as fm, rc
import matplotlib.pyplot as plt
import nlplot
import streamlit.components.v1 as stc
from wordcloud import WordCloud
import collections
import plotly.graph_objs as go
import plotly.offline as offline
# offline.init_notebook_mode(connected=False)
import datetime
from datetime import timedelta
import seaborn as sns
sns.set()
plt.rcParams['font.family'] = "MS Gothic"
import csv
import os



class micro_servis():
    def __init__(self,file):
        doc = docx.Document(file)
        self.file_name = file.name[:-5]
        self.times =[]
        self.names = []
        self.talks = []
        
        for texts in doc.paragraphs:
            self.times.append(texts.text.split("\n")[0])
            self.names.append(texts.text.split("\n")[1])
            self.talks.append(texts.text.split("\n")[2])
        #データフレーム作成
        name = np.array(self.names)
        talk = np.array(self.talks)
        df = pd.DataFrame({"time":self.times})
        for na in np.unique(name):
            na = na.split(" ")
            na = na[2]+na[3]
            df[na] = 0
        for i in range(len(talk)):
            na = name[i].split(" ")
            na = na[2]+na[3]
            df[na][i] = talk[i]
        self.df = df.astype(str)
        st.dataframe(self.df)
        
    def make_dataframe(self):
        self.df.to_csv("csv/{:}.csv".format(self.file_name),encoding='shift_jis',index=False)

    def vis_talk_amount(self):
        df_tmp = self.df.drop(["time"],axis=1).copy()
        df2 = pd.DataFrame({"mo":[0]})
        for i in df_tmp.columns:
            df_kk = df_tmp[df_tmp[i]!="0"]
            df_np = np.array(df_kk).reshape(-1)
            df2[i] = len(df_np)
        df2 = df2.drop(["mo"],axis=1)

        fig = go.Figure()
        trace1 = go.Bar(x=df2.columns, y=np.array(df2.iloc[0,:]))
        
        fig.add_trace(trace1)
        #レイアウトの設定
        fig.update_layout(title='', 
                        width=750,
                        height=500)
        
        #軸の設定
        fig.update_xaxes(title='名前')
        fig.update_yaxes(title='発言量')
        
        # fig.show()
        
        # fig = plt.figure(figsize=(12,9))
        # ax = plt.axes()
        # plt.bar(df2.columns, np.array(df2.iloc[0,:]), label='talk_amount')
        # plt.tick_params(labelsize=18)
        # plt.tight_layout()
        # plt.legend()
        st.write('### 各参加者の発言量')
        st.write(fig)
    
    def vis_wordcloud(self):
        char_filters = [UnicodeNormalizeCharFilter(), #Unicodeを正規化することで表記ゆれを吸収
                        #RegexReplaceCharFilter('<.*?>', '')　#正規表現パターンにマッチした文字列を置換
                        ]
        token_filters = [POSKeepFilter(['名詞']), # 取得するトークンの品詞を指定（今回は名詞）
                        LowerCaseFilter(), #英字を小文字にする
                        #ExtractAttributeFilter('surface'),# トークンから抽出する属性
                        CompoundNounFilter()# 複合語
                        ]
        # a = Analyzer(char_filters=char_filters, token_filters=token_filters)
        a = Analyzer(token_filters=token_filters)
        sentence_list = []
        word_list = []
        ban_word = ["それ","これ","ん","こっち","そう","何","どこ","こと","ここ","何これ","私","俺","オッケー","なん","んん","の","コルタナさん"]
        for i in range(len(self.talks)):
            sentence = []
            for j in a.analyze(self.talks[i]):
                token_list = j.part_of_speech.split(",")
                if token_list[0] == "名詞" and j.surface not in ban_word and 5<=len(j.surface)<=15:
                    sentence.append(j.surface)
                    word_list.append(j.surface)
            sentence_list.append(sentence)
            
        words =  " ".join(word_list)
        wordcloud = WordCloud(background_color = "white",font_path = 'ipaexg.ttf',width = 800,height = 600).generate(words)
        wordcloud.to_file("{:}.png".format(self.file_name))

        st.write('### 会議全体のダイジェスト')
        st.image("{:}.png".format(self.file_name),caption = '会議ダイジェスト',use_column_width = True)
    
    def vis_network(self):
        t = Tokenizer()
        textdf = pd.DataFrame({"text":self.talks})
        textdf["words"] = "a"
        sentence_list = []


        for i in range(len(self.talks)):
            sentence = []
            for j in t.tokenize(self.talks[i]):
                token_list = j.part_of_speech.split(",")
                if token_list[0] == "名詞":
                    sentence.append(j.surface)

            textdf["words"][i] = sentence

        npt = nlplot.NLPlot(textdf, target_col='words')
        stopwords = npt.get_stopword(top_n=0, min_freq=0)

        npt.bar_ngram(
            title='{:}'.format(self.file_name),
            xaxis_label='word_count',
            yaxis_label='word',
            ngram=1,
            top_n=50,
            stopwords=stopwords,
            save=True
        )

        # npt.build_graph(stopwords=stopwords, min_edge_frequency=1)

        # npt.co_network(
        #     title='Co-occurrence network',
        #     save=True
        # )
        dt_now = datetime.datetime.now()
        stc.html('<img width="200" alt="test" src="{:}-{:}-{:}_{:}.html">'.format(dt_now.year,dt_now.month,dt_now.day,self.file_name))
    
    def vis_posinega(self):
        # 発言があるセルのみを抽出
        df = self.df.copy()
        name_list = list(self.df.drop("time",axis=1).columns)
        # name_list.insert(0, "全員")
        
        genre = st.selectbox(
            "発言を見たい人を選んでね！",
            (name_list))
        
        # if genre == "全員":
        #     st.write("全員")
        #     for name in name_list:
        #         df_genre = df[df[name] != 0]
        #         df_genre = df_genre[name]
        #         # インデックスの振りなおし
        #         df_genre = df_genre.reset_index(drop=True)
        #         # 全てのセルの中身を結合して一つの文章にする
        #         genre_fulltext = ""
        #         for i in range(len(df_genre)):
        #             genre_fulltext += df_genre[i]
        #         t = Tokenizer()
        #         s = genre_fulltext
        #         c = collections.Counter(t.tokenize(s, wakati=True))
        #         df_posinega = pd.DataFrame([], columns = name_list)
        #         agree_list = ["うん", "そう","なるほど"]
        #         oppose_list = ["いや","ちがう"]
        #         df_posinega.loc[agree_list[i],name] = c[agree_list[i]]
        # else:
        df_genre = df[df[genre] != 0]
        df_genre = df_genre[genre]
        # インデックスの振りなおし
        df_genre = df_genre.reset_index(drop=True)
        # 全てのセルの中身を結合して一つの文章にする
        genre_fulltext = ""
        for i in range(len(df_genre)):
            genre_fulltext += df_genre[i]

        # Tokenizerで文章の切り分け
        
        t = Tokenizer()
        s = genre_fulltext
        c = collections.Counter(t.tokenize(s, wakati=True))

        # 賛成・反対の単語抽出
        # 賛成を表す単語リスト
        agree_list = ["うん", "そう","なるほど"]
        # 反対を表す単語リスト
        oppose_list = ["いや","ちがう"]

        df_agree = pd.DataFrame([], columns = ["回数","属性"])
        df_oppose = pd.DataFrame([], columns = ["回数","属性"])

        for i in range(len(agree_list)):
            print(agree_list[i], c[agree_list[i]])
            df_agree.loc[agree_list[i],"回数"] = c[agree_list[i]]
            df_agree.loc[agree_list[i],"属性"] = "賛成"

        for i in range(len(oppose_list)):
            print(oppose_list[i], c[oppose_list[i]])
            df_oppose.loc[oppose_list[i],"回数"] = c[oppose_list[i]]
            df_oppose.loc[oppose_list[i],"属性"] = "反対"

        df_graph = pd.concat([df_agree,df_oppose])

        # 属性（賛成/反対）に応じて色を割り振る
        attribute_list = list(df_graph['属性'])
        color_list = []

        for i in range(len(attribute_list)):
            if attribute_list[i] == '賛成':
                color_list.append('rgba(222,45,38,0.8)')
            else:
                color_list.append('rgba(131, 203, 250,0.8)')
        print(df_graph)
        trace0 = go.Bar(
            x=df_graph.index,
            y=df_graph['回数'],
            marker=dict(
                color=color_list),
        )

        data = [trace0]
        layout = go.Layout(
            title='賛成・反対ワードの発言数',
        )

        fig = go.Figure(data=data, layout=layout)
        # py.iplot(fig, filename='color-bar')
        #offline.iplot(fig)
        st.write('### {:}が発言した賛成・反対ワード'.format(genre))
        st.write(fig)

        df_q = pd.DataFrame([], columns = ["回数"])

        print('？', c['？'])
        df_q.loc['？'] = c['？']

        st.write('### {:}が行った質問・問いかけ'.format(genre))
        st.write('{:}が質問・問いかけを行った回数は'.format(genre),c['？'],'回でした！')
        
    # @st.cache()
    def vis_silence(self):
        
        df = self.df.copy()
        name_list = self.df.drop("time",axis=1).columns
        starttime = []
        endtime = []
        for i in range(len(df["time"])-1):
            starttime.append(pd.to_datetime(df["time"].str[:12][i]))
            endtime.append(pd.to_datetime(df["time"].str[-12:][i]))

        timedf = pd.DataFrame({"starttime":starttime,"endtime":endtime})

        diff_time = []
        for i in range(len(timedf)):
            diff_time.append(timedf["endtime"][i].timestamp()-timedf["starttime"][i].timestamp())
        timedf["diff"] = diff_time

        pass_list = []
        pass_time = 0
        for i in range(len(timedf)):
            pass_time = timedf["endtime"][i].timestamp()-timedf["starttime"][0].timestamp()
            pass_list.append(pass_time)
        timedf["pass"] = pass_list

        for i in name_list:
            timedf[i] = df[i]

        cumsum_ = []
        for n in range(1,10000):
            k = 600*n
            if k>float(timedf["pass"][-1:]):
                k = float(timedf["pass"][-1:])
                
            glo_amount = 0
            for i in name_list:
                timeindex = timedf[(timedf[i]!="0") & (timedf["pass"]<k) & (timedf["pass"]>=600*(n-1))]
                if timedf[i][timeindex.index].sum() != 0:
                    glo_amount += len(timedf[i][timeindex.index].sum())
            
            for i in name_list:
                timeindex = timedf[(timedf[i]!="0") & (timedf["pass"]<k) & (timedf["pass"]>=600*(n-1))]
                if timedf[i][timeindex.index].sum() != 0:
                    val = len(timedf[i][timeindex.index].sum())/glo_amount
                    cumsum_.append(val)
                else:
                    cumsum_.append(0)
            
            if k==float(timedf["pass"][-1:]):
                break
            
        cumsum_df = pd.DataFrame(np.array(cumsum_).reshape(-1,len(name_list)))
        cumsum_df.columns=name_list
        select_name = list(name_list)
        select_name.insert(0, "全員")
        
        genre = st.selectbox(
            "発言を見たい人を選んでね！",
            (select_name))
        
        st.write('### 会話の支配度')
        fig = go.Figure()
        
        if genre=="全員":
            for i in name_list:
                fig.add_trace(go.Scatter(x=np.array((range(len(cumsum_df))))*10,
                         y=np.array(cumsum_df[i]),
                         mode='lines',
                         name=i,
                        ),
                )
        else:
            fig.add_trace(go.Scatter(x=np.array(range(len(cumsum_df)))*10,
                        y=np.array(cumsum_df[genre]),
                        mode='lines',
                        name=genre,
                    ),
            )
        
        fig.update_xaxes(title='時間(分)')
        fig.update_yaxes(title='話者の割合')
        fig.update_layout(title='', 
                        width=750,
                        height=500)
        st.write(fig)

    def vis_repeat_count(self):
        df2 = pd.DataFrame({"発言者":self.names, "内容":self.talks})

        change = []
        for num in range(len(df2["発言者"])-1):
            if df2["発言者"][num+1] != df2["発言者"][num]:
                change.append(num+1)

        t = Tokenizer()

        for i in range(len(self.talks)):
            sentence = []
            for j in t.tokenize(self.talks[i]):
                token_list = j.part_of_speech.split(",")
                if token_list[0] == "名詞":
                    if token_list[1] != "非自立":
                        sentence.append(j.surface)
                elif token_list[0] == "動詞":
                    if token_list[1] == "自立":
                        sentence.append(j.surface)
                elif token_list[0] == "形容詞":
                    sentence.append(j.surface)

            df2["内容"][i] = sentence


        uni_names = np.unique(self.names)

        repeater = []
        for i in change:
            for j in range(len(df2["内容"][i])):
                if df2["内容"][i][j] in df2["内容"][i-1]:
                    repeater.append(df2.loc[i, "発言者"])          
                    break

        repeatcount = []
        talker = []
        talkercount = []
        
        for i in range(len(df2["発言者"])):
            talker.append(df2.loc[i, "発言者"])

        for i in uni_names:
            repeatcount.append(repeater.count(i))   
            talkercount.append(talker.count(i)) 
    
    
        # print(repeatcount/talkercount)
        repeatratio = [x*100 / y for (x, y) in zip(repeatcount, talkercount)]

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.bar(self.df.drop(["time"],axis=1).columns, repeatcount, color = "orange", align="edge", width=-0.3, label='repeat_count') #label='repeat_count'
        ax2.bar(self.df.drop(["time"],axis=1).columns, repeatratio, align="edge", width=0.3, label='repeat_ratio')
        ax1.set_ylabel("反復した回数")  #y1軸ラベル
        ax2.set_ylabel("反復した割合[%]")  #y2軸ラベル
        plt.tick_params(labelsize=10)
        ax1.tick_params(labelsize=10)
        plt.tight_layout()
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2 ,loc='best')
        plt.grid(False)
        st.write('### 各参加者の傾聴回数')
        st.pyplot(fig)

    def Q_category(self):
        # トランスクリプションのデータフレームをコピーしてくる
        df = self.df.copy()
        name_list = list(self.df.drop("time",axis=1).columns)
        # name_list.insert(0, "全員")
        
        genre = st.selectbox(
            "発言を見たい人を選んでね！",
            (name_list))
        
        df_genre = df[df[genre] != 0]
        df_genre = df_genre[genre]
        # インデックスの振りなおし
        df_genre = df_genre.reset_index(drop=True)

        st.dataframe(df_genre)

        df_q = df_genre[df_genre.str.contains('？')]

        df_q1 = pd.DataFrame(df_q)
        # インデックスの振りなおし
        df_q1 = df_q1.reset_index(drop=True)
        df_q1['Label'] = '0'

        str_l = df_q1[genre]
        mydict = {"Why":0, "What":0, "When":0, "Where":0, "Who":0, "How":0}
        for i in range(len(str_l)):
            if 'なんで' in str_l[i]:
                print(i)
                df_q1['Label'][i] = 'Why'
                mydict['Why'] += 1
            if '何' in str_l[i]:
                df_q1['Label'][i] = 'What'
                mydict['What'] += 1
            if 'いつ' in str_l[i]:
                df_q1['Label'][i] = 'When'
                mydict['When'] += 1
            if 'どこ' in str_l[i]:
                df_q1['Label'][i] = 'Where'
                mydict['Where'] += 1
            if 'だれ' in str_l[i]:
                df_q1['Label'][i] = 'Who'
                mydict['Who'] += 1
            if 'どう' in str_l[i]:
                df_q1['Label'][i] = 'How'
                mydict['How'] += 1
        
        st.dataframe(df_q1)
        st.write(mydict)

        data = go.Bar(x=list(mydict.keys()), y=list(mydict.values()))
        fig = go.Figure(data)
        st.write(fig)
        



st.write('# 会議からあなたの毎日を変える　Fromee')

# if os.path.isfile("https://share.streamlit.io/hiroyoungkun/streamlitapps/main/test.csv") == True:
#     with open('https://share.streamlit.io/hiroyoungkun/streamlitapps/main/test.csv', 'w', newline="") as f:
#         printf("fuck")
#         writer = csv.writer(f)
#         writer.writerow(['満足度', '会議の種類', '予定時間'])

# df1 = pd.read_csv("https://share.streamlit.io/hiroyoungkun/streamlitapps/main/test.csv")
df1 = pd.read_csv("https://github.com/HiroYoungKun/StreamlitApps/main/Sample-Spreadsheet-10-rows.csv")
# print(df1)
st.write(df1)
satis = st.radio(
    "この会議の満足度は？",
    ('1', '2', '3', '4', '5')
    )


max_value = 200
min_value = 0

settime = st.slider("予定していた設定時間（分）",min_value,max_value,30,10)

type = st.selectbox(
    "会議の種類を選択してください",
    ('情報共有や周知のための話し合い', 'アイデアを創造したり問題解決をするための話し合い', '役割や利害を調整するための話し合い', '意思決定を行うための話し合い', '勉強会や研修など教育・指導のための話し合い')
)


if st.button('確定'):
    df2 = pd.DataFrame({"満足度" : [satis],"予定時間" : [settime],"会議の種類" : [type]},)
    dfmain = pd.concat([df1,df2])
    dfmain.to_csv(("https://share.streamlit.io/hiroyoungkun/streamlitapps/main/test.csv"),index=False)
else:
    st.write('確定ボタンを押してください')

# Streamlitによるファイルアップロード機能
uploaded_file=st.file_uploader("↓Teamsのトランスクリプトファイル（docx）をアップロード↓", type=['docx'])
# 受け取ったワードファイルをワード形式で読み込んでdocという変数に渡す

#トランスクリプトの読み込み
#/nごとにわけて時間、名前、内容のリストを作成

debug = False

if uploaded_file is None:
    st.write('## ファイルをアップしてください！')

else:
    mi = micro_servis(uploaded_file)
    if debug:
        # mi.make_dataframe()
        st.write("debug")
        mi.vis_posinega()
    else:
        # mi.make_dataframe()
        add_selectbox = st.sidebar.selectbox(
        "見たいグラフを選んでね",
        ("ダイジェスト", "発言量", "ポジネガ発言", "会話の活性度","問いかけの種類と回数","yeeeees")
        )
        
        if add_selectbox == "ダイジェスト":
            mi.vis_wordcloud()
        if add_selectbox == "発言量":
            mi.vis_talk_amount()
        if add_selectbox == "ネットワーク":
            mi.vis_network()
        if add_selectbox == "ポジネガ発言":
            mi.vis_posinega()
        if add_selectbox == "会話の活性度":
            mi.vis_silence()
        if add_selectbox == "問いかけの種類と回数":
            mi.Q_category()
