import streamlit as st
import pandas as pd
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import re



st.set_page_config(
    page_title="DecomAI", 
    page_icon = "logo-transparent.png",
    layout="wide")

st.image("decom_logo.png", width=120)
# with st.sidebar:
#     st.image("decom_logo.png", width=120)
st.title("DecomAI")


with st.sidebar:
    api_key = st.text_input("OpenAI APIキー🔑") 


st.subheader('ファイルアップロード')

data = st.file_uploader("CSVファイルをアップロードしてください", type='csv') # ファイルアップロード
if data is not None:
    df = pd.read_csv(data)
    st.text("アップロードしたデータ")
    st.write(df)
    cols = tuple(df.columns)
    st.subheader('対象の列選択')
    target_cols = st.multiselect("対象の列を選択してください(複数選択可)", cols) # セレクトボックス

st.subheader('①４要素要約')

prompt_text = st.text_area(
    "４要素に要約させるためのプロンプトを入力してください",
    '''あなたは優秀なアシスタントです。
以下のアンケートへの回答において、回答者についてシーン（場面）・ドライバー（厳選要因）・エモーション（情緒）・バックグラウンド（背景要因）を整理してそれぞれ回答してください。
それぞれ必ず改行で区切って、それぞれ文章で記述してください。''') # 文字入力(複数行)





def find_text_start_from(keyword,text):
   search = keyword +".+"
   result = re.search(search, text)
   if result == None:
       return None
   else:
       return result.group(0).replace(keyword,"").strip()

# 4要素要約
def free_text2sumups_a(free_text):
    # ChatGPTに聞く
    output = chatgpt_chain.predict(human_input=free_text) 
    
    # 回答を４つに分ける
    scene = find_text_start_from('シーン：', output)
    driver = find_text_start_from('ドライバー：', output)
    emotion = find_text_start_from('エモーション：', output)
    background = find_text_start_from('バックグラウンド：', output)
    
    return [scene, driver, emotion, background]

# 対象カラムの内容を要約してdfで出力
def target_col2newdf_a(target_col):
    scenes = []
    drivers = []
    emotions = []
    backgrounds = []
    for free_text in df[target_col]:
        print(free_text)
        sumups = free_text2sumups_a(free_text)
        # 実行カウント
        global progress_count
        progress_count = progress_count + 1
        my_bar.progress(progress_count/progress_total, text=progress_text)

        scenes.append(sumups[0])
        drivers.append(sumups[1])
        emotions.append(sumups[2])
        backgrounds.append(sumups[3])

    newlists = list(zip(scenes, drivers, emotions, backgrounds))
    newdf = pd.DataFrame(newlists, columns=[f'{target_col}_scenes', f'{target_col}_drivers', f'{target_col}_emotions', f'{target_col}_backgrounds'])
    return newdf
progress_count = 0
progress_total = 0
if st.button("４要素に要約",key='sumup'):
    if api_key == '':
        st.error('APIキーを入力してください', icon="🚨")
    if data is None:
        st.error('先にファイルをアップロードしてください', icon="🚨")
    
    if api_key != '' and data is not None:
        template = prompt_text+"""
        Human: {human_input}
        Assistant:"""
        prompt = PromptTemplate(
            input_variables=["human_input"], 
            template=template
        )

        chatgpt_chain = LLMChain(
            llm=OpenAI(temperature=0, openai_api_key=api_key), 
            prompt=prompt, 
            verbose=True
        )

        # プログレスバー
        progress_text = "４要素要約を実行中...処理が終わるまでしばらくお待ちください。"
        progress_total = len(target_cols) * df.shape[0]
        my_bar = st.progress(0, text=progress_text)
        

            
        # 全てのカラムに対して処理
        for target_col in target_cols:
            newdf = target_col2newdf_a(target_col)
            st.info(f'{target_col}について要約完了')
            df = pd.concat([df, newdf], axis=1) 
        st.success('要約完了')
        st.text("４要素要約済みデータ")
        st.write(df)
        st.session_state['df'] = df


st.subheader('②エモーション深掘り')

# emotion深掘り
def free_text2sumups_b(free_text, emotion_pre):
    # ChatGPTに聞く
    output = chatgpt_chain_b.predict(emotion=emotion_pre, human_input=free_text) 
    return output


# 対象カラムの内容を要約してdfで出力
def target_col2newdf_b(target_col):
    emotion_deepdive = []
    for index, row in df[[target_col, target_col+'_emotions']].iterrows():
        free_text = row[0]
        emotion_pre = row[1]
        print(free_text)
        sumups = free_text2sumups_b(free_text, emotion_pre)
        # 実行カウント
        global progress_count
        progress_count = progress_count + 1
        my_bar.progress(progress_count/progress_total, text=progress_text)

        emotion_deepdive.append(sumups)

    newlists = emotion_deepdive
    newdf = pd.DataFrame(newlists, columns=[f'{target_col}_emotionDeepdive'])
    return newdf



prompt_text = st.text_area(
    "エモーションの要約を元に深掘りするためのプロンプトを入力してください",
    '''あなたは優秀なアシスタントです。
以下のアンケートの回答において、回答者が{emotion}と感じている理由と裏にある価値観を教えてください。''') # 文字入力(複数行)


if st.button("エモーションの要約を元に深掘り", key='emotion'):
    
    if api_key is '':
        st.error('APIキーを入力してください', icon="🚨")
    if data is None:
        st.error('先にファイルをアップロードしてください', icon="🚨")
    
    if api_key != '' and data is not None:
        template = prompt_text+"""
        Human: {human_input}
        Assistant:"""
        prompt = PromptTemplate(
            input_variables=["emotion", "human_input"], 
            template=template
        )

        chatgpt_chain_b = LLMChain(
            llm=OpenAI(temperature=0, openai_api_key=api_key), 
            prompt=prompt, 
            verbose=True
        )

        # プログレスバー
        progress_text = "エモーション深掘りを実行中...処理が終わるまでしばらくお待ちください。"
        progress_total = len(target_cols) * df.shape[0]
        my_bar = st.progress(0, text=progress_text)
        
        df = st.session_state['df']
            
        # 全てのカラムに対して処理
        for target_col in target_cols:
            # st.write(df.columns)
            newdf = target_col2newdf_b(target_col)
            st.info(f'{target_col}についてエモーション深掘り完了')
            df = pd.concat([df, newdf], axis=1) 
        st.success('要約完了')
        st.text("エモーション深掘り済みデータ")
        st.write(df)
        st.session_state['df'] = df
        output_csv = df.to_csv('output.csv', encoding="utf-8")
        st.session_state['output_csv'] = df.to_csv('output.csv')


        st.subheader('処理済みCSVファイルダウンロード')
        # with open(st.session_state['output_csv']) as f:
        #     st.download_button('Download CSV', f)    
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')


        csv = convert_df(st.session_state['df'])

        st.download_button(
        "Press to Download",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
        )
        # st.success('ダウンロード完了')

# st.multiselect("multiselectbox", ("select1", "select2")) # 複数選択可能なセレクトボックス
# st.radio("radiobutton", ("radio1", "radio2")) # ラジオボタン
# st.text_input("text input") # 文字入力(1行)
# st.text_area("text area") # 文字入力(複数行)
# st.slider("slider", 0, 100, 50) # スライダー

# import streamlit as st

# # Using object notation
# add_selectbox = st.sidebar.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone")
# )

# # Using "with" notation
# with st.sidebar:
#     add_radio = st.radio(
#         "Choose a shipping method",
#         ("Standard (5-15 days)", "Express (2-5 days)")
#     )



# 3. 価値を端的に表現


st.subheader('③価値を端的に表現')


# # 対象カラムの内容を要約してdfで出力
# def target_col2newdf_3(target_col):
#     emotion_deepdive = []
#     for index, row in df[[target_col, target_col+'_emotions']].iterrows():
#         free_text = row[0]
#         emotion_pre = row[1]
#         print(free_text)
#         sumups = free_text2sumups_1(free_text, emotion_pre)
#         # 実行カウント
#         global progress_count
#         progress_count = progress_count + 1
#         my_bar.progress(progress_count/progress_total, text=progress_text)

#         emotion_deepdive.append(sumups)

#     newlists = emotion_deepdive
#     newdf = pd.DataFrame(newlists, columns=[f'{target_col}_value'])
#     return newdf
prompt_text = st.text_area(
    "価値を端的に表現するためのプロンプトを入力してください",
    '''あなたは優秀なアシスタントです。
では、この人物がこのエピソードに感じている価値を端的に表現してみて。なぜこのことが気に入っているのか、端的に表現してみて。
''') # 文字入力(複数行)

# if st.button("価値を端的に表現", key='value'):
#     test_apikey()
#     df = st.session_state['df']
#     if api_key != '' and data is not None:
#         template = prompt_text+"""
#         Human: {human_input}
#         Assistant:"""
#         prompt = PromptTemplate(
#             input_variables=["human_input"], 
#             template=template
#         )

#         chatgpt_chain_b = LLMChain(
#             llm=OpenAI(temperature=0, openai_api_key=api_key, model_name=model_name), 
#             prompt=prompt, 
#             verbose=True
#         )

#         # プログレスバー
#         progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
#         progress_total = len(target_cols) * df.shape[0]
#         my_bar = st.progress(0, text=progress_text)
        
#         df = st.session_state['df']
            
#         # 全てのカラムに対して処理
#         for target_col in target_cols:
#             # st.write(df.columns)
#             newdf = target_col2newdf_3(target_col)
#             st.info(f'{target_col}について価値を端的に表現完了')
#             df = pd.concat([df, newdf], axis=1) 
#         st.success('完了')
#         st.text("価値を端的に表現済みデータ")
#         st.write(df)
#         st.session_state['df'] = df


# # 対象カラムの内容を要約してdfで出力
# def target_col2newdf_0(target_col, newColName, usingColName, prompt_text): # NewColNameに"_emotion"みたいに引用する列の名前を入れる
#     newlists = []
#     for index, row in df[[target_col, target_col+usingColName]].iterrows():
#         human_input = row[0]
#         emotion_pre = row[1]
#         print(human_input)
#         sumups = free_text2sumups_1(attributesInfo, prompt_text, human_input, emotion_pre)
#         # 実行カウント
#         global progress_count
#         progress_count = progress_count + 1
#         my_bar.progress(progress_count/progress_total, text=progress_text)

#         newlists.append(sumups)

#     newdf = pd.DataFrame(newlists, columns=[f'{target_col}_{newColName}'])
#     return newdf


if st.button("価値を端的に表現", key='value'):
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)

    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = target_col2newdf_1(target_col, newColName="simpleValue", usingColName="", prompt_text=prompt_text)
        st.info(f'{target_col}について完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("価値を端的に表現済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")





# 4. 価値の深掘り
# 対象カラムの内容を要約してdfで出力
def target_col2newdf_4(target_col):
    emotion_deepdive = []
    for index, row in df[[target_col, target_col+'_value']].iterrows():
        free_text = row[0]
        value = row[1]
        print(free_text)
        sumups = free_text2sumups_b(free_text, value)
        # 実行カウント
        global progress_count
        progress_count = progress_count + 1
        my_bar.progress(progress_count/progress_total, text=progress_text)

        emotion_deepdive.append(sumups)

    newlists = emotion_deepdive
    newdf = pd.DataFrame(newlists, columns=[f'{target_col}_valueDeepdive'])
    return newdf


st.subheader('④価値の深掘り')
prompt_text = st.text_area(
    "価値を深掘りするためのプロンプトを入力してください",
    '''この人が{value}を求めているのは、なぜでしょうか。 この人の価値観を端的に表現してみて。
''') # 文字入力(複数行)

if st.button("価値を端的に表現", key='value_deepdive'):
    if api_key is '':
        st.error('APIキーを入力してください', icon="🚨")
    if data is None:
        st.error('先にファイルをアップロードしてください', icon="🚨")
    if api_key != '' and data is not None:
        template = prompt_text+"""
        Human: {human_input}
        Assistant:"""
        prompt = PromptTemplate(
            input_variables=["human_input"], 
            template=template
        )

        chatgpt_chain_b = LLMChain(
            llm=OpenAI(temperature=0, openai_api_key=api_key), 
            prompt=prompt, 
            verbose=True
        )

        # プログレスバー
        progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
        progress_total = len(target_cols) * df.shape[0]
        my_bar = st.progress(0, text=progress_text)
        
        df = st.session_state['df']
            
        # 全てのカラムに対して処理
        for target_col in target_cols:
            # st.write(df.columns)
            newdf = target_col2newdf_4(target_col)
            st.info(f'{target_col}について価値を端的に表現完了')
            df = pd.concat([df, newdf], axis=1) 
        st.success('完了')
        st.text("価値を端的に表現済みデータ")
        st.write(df)
        st.session_state['df'] = df


# 5. 不満・未充足
# 対象カラムの内容を要約してdfで出力
def target_col2newdf_5(target_col):
    emotion_deepdive = []
    for index, row in df[[target_col, target_col+'_complain']].iterrows():
        free_text = row[0]
        emotion_pre = row[1]
        print(free_text)
        sumups = free_text2sumups_b(free_text, emotion_pre)
        # 実行カウント
        global progress_count
        progress_count = progress_count + 1
        my_bar.progress(progress_count/progress_total, text=progress_text)

        emotion_deepdive.append(sumups)

    newlists = emotion_deepdive
    newdf = pd.DataFrame(newlists, columns=[f'{target_col}_valueDeepdive'])
    return newdf


st.subheader('⑤不満・未充足')
prompt_text = st.text_area(
    "不満・身充足を明らかにするためのプロンプトを入力してください",
    '''{value}において、{category}で解消することができる、いまの消費者の隠れた不満や未充足欲求は何ですか？
''') # 文字入力(複数行)

if st.button("不満・未充足を明らかにする", key='complain'):
    if api_key is '':
        st.error('APIキーを入力してください', icon="🚨")
    if data is None:
        st.error('先にファイルをアップロードしてください', icon="🚨")
    if api_key != '' and data is not None:
        template = prompt_text+"""
        Human: {human_input}
        Assistant:"""
        prompt = PromptTemplate(
            input_variables=["human_input"], 
            template=template
        )

        chatgpt_chain_b = LLMChain(
            llm=OpenAI(temperature=0, openai_api_key=api_key), 
            prompt=prompt, 
            verbose=True
        )

        # プログレスバー
        progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
        progress_total = len(target_cols) * df.shape[0]
        my_bar = st.progress(0, text=progress_text)
        
        df = st.session_state['df']
            
        # 全てのカラムに対して処理
        for target_col in target_cols:
            # st.write(df.columns)
            newdf = target_col2newdf_4(target_col)
            st.info(f'{target_col}について価値を端的に表現完了')
            df = pd.concat([df, newdf], axis=1) 
        st.success('完了')
        st.text("価値を端的に表現済みデータ")
        st.write(df)
        st.session_state['df'] = df


# 6. 社会潮流・消費者トレンド
st.subheader('⑥社会潮流・消費者トレンド')
prompt_text = st.text_area(
    "社会潮流・消費者トレンドを明らかにするためのプロンプトを入力してください",
    '''{value}が、いま求められている背景には、どんな「社会潮流」や「消費者トレンド」がありますか？３つ挙げてください
''')
st.button("社会潮流・消費者トレンドを明らかにする", key='trend')
# 7. 新商品コンセプト
st.subheader('⑦新商品コンセプト')
prompt_text = st.text_area(
    "新商品コンセプトを考えるためのプロンプトを入力してください",
    '''この人が欲しがりそうな{category}の新商品コンセプトを200文字で作成してください
''')
st.button("新商品コンセプトを考える", key='concept')
# 8. キャッチコピー案
st.subheader('⑧キャッチコピー案')
prompt_text = st.text_area(
    "キャッチコピー案を考えるためのプロンプトを入力してください",
    '''あなたは、日本の著名なコピーライターです。 新商品「{value}」の広告キャッチコピーを考えてみて。
''')
st.button("キャッチコピー案を考える", key='copy')
# 9. 販売ルート
st.subheader('⑨販売ルート')
prompt_text = st.text_area(
    "販売ルートを考えるためのプロンプトを入力してください",
    '''新商品「{value}」の販売ルートを５つ考えてみてください
''')
st.button("販売ルートを考える", key='route')
# 10. SoB
st.subheader('⑩SoB')
prompt_text = st.text_area(
    "SoBを考えるためのプロンプトを入力してください",
    '''いま消費者が求めている欲求のひとつに、「{value}」が挙げられます。
この欲求を充たしいている商品やサービスを５つ挙げてください。
''')
st.button("SoBを考える", key='sob')

st.button("全て実行する", key='all')


