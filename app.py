import streamlit as st
import pandas as pd
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import re



st.set_page_config(
    page_title="DecomAI", 
    page_icon = "decom_logo.png",
    layout="wide")

st.image("/app/chatgpt_automation_test/decom_logo.png", width=120)
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
