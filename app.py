# googleへの保存を試す
import streamlit as st
import pandas as pd
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import re
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

JSON_FILE = "service_account.json"
ID = "1DBQd9ZuF5VmB5IObmuzultptmF2TUwU-"

gauth = GoogleAuth()
scope = ["https://www.googleapis.com/auth/drive"]
gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(JSON_FILE, scope)
drive = GoogleDrive(gauth)



def savefile(df, file_name):
    now = datetime.now()
    time_str = now.strftime("temp-%Y%m%d-%H%M")
    file = drive.CreateFile({"title": time_str+file_name+".csv", "parents": [{"id": ID}]})
    # file.SetContentString("テスト")
    df.to_csv(time_str+"-"+file_name+'.csv', index=False)
    file.SetContentFile(time_str+"-"+file_name+'.csv')
    file.Upload()
def savefinalfile(df, file_name):
    now = datetime.now()
    time_str = now.strftime("FINAL-%Y%m%d-%H%M")
    file = drive.CreateFile({"title": time_str+"-"+file_name+".csv", "parents": [{"id": ID}]})
    # file.SetContentString("テスト")
    df.to_csv(time_str+"-"+file_name+'.csv', index=False)
    file.SetContentFile(time_str+"-"+file_name+'.csv')
    file.Upload()




st.set_page_config(
    page_title="DecomAI", 
    page_icon = "/Users/kotaro/Documents/chatgpt-automation/decom_logo.png",
    layout="wide")
st.image("/Users/kotaro/Documents/chatgpt-automation/decom_logo.png", width=120)
st.title("DecomAI")
with st.sidebar:
    api_key = st.text_input("OpenAI APIキー🔑") 
api_key = "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"
category = st.text_input("商品カテゴリーを入力してください") 
# category = "test"

# 使うモデルを選択
options = ["gpt-4", "gpt-3.5-turbo"]
model_name = st.selectbox('利用するモデルを選んでください', options)
model_name = "gpt-3.5-turbo"

st.subheader('ファイルアップロード')

data = st.file_uploader("CSVファイルをアップロードしてください", type='csv') # ファイルアップロード
if data is not None:
    df = pd.read_csv(data)
    st.text("アップロードしたデータ")
    st.write(df)
    df['category'] = category
    cols = tuple(df.columns)
    st.subheader('対象の列選択')
    target_cols = st.multiselect("対象の列を選択してください(複数選択可能)", cols) # セレクトボックス
    # target_cols = ["Q1_1"]

    # 属性情報選択
    st.subheader('⓪属性情報')
    target_cols_attributes = st.selectbox("属性情報を表す列を選択してください(1つのみ選択可)", cols) # セレクトボックス

# API Keyが入力されているか確認
def test_apikey():
    if api_key == '':
        st.error('APIキーを入力してください', icon="🚨")
    if data is None:
        st.error('先にファイルをアップロードしてください', icon="🚨")















progress_count = 0
progress_total = 0
# 属性情報、プロンプトテキスト、[その他の参照するカラムのリスト]を受け取って、GPTの返答テキストを返す
def ask_gpt(attributesInfo, prompt_text,human_input, addons):
    template = prompt_text+"""
回答者の属性情報: {attributesInfo}
回答: {human_input}
Assistant:
        """
    if len(addons) == 0:
        template = prompt_text+"""
        回答者の属性情報: {attributesInfo}
        回答: {human_input}
        Assistant:
        """
        prompt = PromptTemplate(
            input_variables=["attributesInfo", "human_input"],
            template=template
        )
        print(prompt)
        chatgpt_chain = LLMChain(
            llm=OpenAI(temperature=0, openai_api_key=api_key, model_name=model_name), 
            prompt=prompt, 
            verbose=True
        )
        output = chatgpt_chain.predict(attributesInfo=attributesInfo, human_input=human_input) 
        print(output)
    elif len(addons) == 1:
        pass
        for key, value in addons.items():
            exec(f"{key} = '{value}'")
        key1 = list(addons.keys())[0]
        value1 = list(addons.items())[0]
        if key1 == 'emotion':
            prompt = PromptTemplate(
                input_variables=["attributesInfo", "human_input", "emotion"],
                template=template
                )
            print(prompt)
            chatgpt_chain = LLMChain(
                llm=OpenAI(temperature=0, openai_api_key=api_key, model_name=model_name), 
                prompt=prompt, 
                verbose=True
            )
            output = chatgpt_chain.predict(attributesInfo=attributesInfo, human_input=human_input, emotion=addons['emotion'])
            print(output)
        elif key1 == "category":
            prompt = PromptTemplate(
                input_variables=["attributesInfo", "human_input", "category"],
                template=template
                )
            print(prompt)
            chatgpt_chain = LLMChain(
                llm=OpenAI(temperature=0, openai_api_key=api_key, model_name=model_name), 
                prompt=prompt, 
                verbose=True
            )
            output = chatgpt_chain.predict(attributesInfo=attributesInfo, human_input=human_input, category=addons['category'])
            print(output)
        elif key1 == "value":
            prompt = PromptTemplate(
                input_variables=["attributesInfo", "human_input", "value"],
                template=template
                )
            print(prompt)
            chatgpt_chain = LLMChain(
                llm=OpenAI(temperature=0, openai_api_key=api_key, model_name=model_name), 
                prompt=prompt, 
                verbose=True
            )
            output = chatgpt_chain.predict(attributesInfo=attributesInfo, human_input=human_input, value=addons['value'])
            print(output)
    elif len(addons) == 2:
        # 追加情報がある場合、内容を変数として定義する
        for key, value in addons.items():
            exec(f"{key} = '{value}'")
        key1 = list(addons.keys())[0]
        value1 = list(addons.items())[0]
        key2 = list(addons.keys())[1]
        value2 = list(addons.items())[1]
        if set([key1, key2]) == set(['emotion', 'value']):
            prompt = PromptTemplate(
                input_variables=["attributesInfo", "human_input", "emotion", "value"],
                template=template
                )
            print(prompt)
            chatgpt_chain = LLMChain(
                llm=OpenAI(temperature=0, openai_api_key=api_key, model_name=model_name), 
                prompt=prompt, 
                verbose=True
            )
            output = chatgpt_chain.predict(attributesInfo=attributesInfo, human_input=human_input, emotion=addons['emotion'], value=addons['value'])
            print(output)
        elif set([key1, key2]) == set(['category', 'value']):
            prompt = PromptTemplate(
                input_variables=["attributesInfo", "human_input", "category", "value"],
                template=template
                )
            print(prompt)
            chatgpt_chain = LLMChain(
                llm=OpenAI(temperature=0, openai_api_key=api_key, model_name=model_name), 
                prompt=prompt, 
                verbose=True
            )
            output = chatgpt_chain.predict(attributesInfo=attributesInfo, human_input=human_input, category=addons['category'], value=addons['value'])
            print(output)
        else:
            st.error('addons読み込みで想定外が起きました', icon="🚨")
            print('error!!!!!!!!')

    else:
        st.error('3つ以上の追加情報をGPTへの質問に利用しようとしています', icon="🚨")
        print('3つ以上')

    return output

# 対象カラム名、新しいカラム名、(属性情報)、プロンプトテキスト、利用カラムリスト（0-2こ）　を受け取って dfを返す
# using_colはリスト
def make_answer_df(target_col, new_col_name, prompt_text, using_col):
    newlists = []
    global progress_count

    if len(using_col) == 0:
        for row in zip(df[target_cols_attributes], df[target_col]):
            attributesInfo = row[0]
            human_input = row[1]
            addons = {}
            sumups = ask_gpt(attributesInfo, prompt_text, human_input, addons)
            # 実行カウント
            # global progress_count
            progress_count = progress_count + 1
            my_bar.progress(progress_count/progress_total, text=progress_text)

            newlists.append(sumups)
    elif len(using_col) == 1:
        if using_col == ['emotion']:
            for row in zip(df[target_cols_attributes], df[target_col], df[target_col+'_emotion']):
                attributesInfo = row[0]
                human_input = row[1]
                addons = {"emotion":row[2]}
                sumups = ask_gpt(attributesInfo, prompt_text, human_input, addons)
                # 実行カウント
                # global progress_count
                progress_count = progress_count + 1
                my_bar.progress(progress_count/progress_total, text=progress_text)

                newlists.append(sumups)
        elif using_col == ['category']:
            for row in zip(df[target_cols_attributes], df[target_col], df['category']):
                attributesInfo = row[0]
                human_input = row[1]
                addons = {"category":row[2]}
                sumups = ask_gpt(attributesInfo, prompt_text, human_input, addons)
                # 実行カウント
                # global progress_count
                progress_count = progress_count + 1
                my_bar.progress(progress_count/progress_total, text=progress_text)

                newlists.append(sumups)
        elif using_col == ['value']:
            for row in zip(df[target_cols_attributes], df[target_col], df[target_col+'_value']):
                attributesInfo = row[0]
                human_input = row[1]
                addons = {"value":row[2]}
                sumups = ask_gpt(attributesInfo, prompt_text, human_input, addons)
                # 実行カウント
                # global progress_count
                progress_count = progress_count + 1
                my_bar.progress(progress_count/progress_total, text=progress_text)

                newlists.append(sumups)
        else:
            print("error!!")
    elif len(using_col) == 2:
        if set(using_col) == set(['category','value']):
            for row in zip(df[target_cols_attributes], df[target_col], df['category'], df[target_col+'_value']):
                attributesInfo = row[0]
                human_input = row[1]
                addons = {"category":row[2],"value":row[3]}
                sumups = ask_gpt(attributesInfo, prompt_text, human_input, addons)
                # 実行カウント
                # global progress_count
                progress_count = progress_count + 1
                my_bar.progress(progress_count/progress_total, text=progress_text)

                newlists.append(sumups)
        elif set(using_col) == set(['emotion','value']):
            for row in zip(df[target_cols_attributes], df[target_col], df[target_col+'_emotion'], df[target_col+'_value']):
                attributesInfo = row[0]
                human_input = row[1]
                addons = {"emotion":row[2],"value":row[3]}
                sumups = ask_gpt(attributesInfo, prompt_text, human_input, addons)
                # 実行カウント
                # global progress_count
                progress_count = progress_count + 1
                my_bar.progress(progress_count/progress_total, text=progress_text)

        
    else:
        st.error('addons読み込みで想定外が起きました', icon="🚨")
        print('error!!!!!!!!')
    
    newdf = pd.DataFrame(newlists, columns=[f'{target_col}_{new_col_name}'])
    return newdf
def find_text_start_from(keyword,text):
   search = keyword +".+"
   result = re.search(search, text)
   if result == None:
       return None
   else:
       return result.group(0).replace(keyword,"").strip()

# 1. 4要素要約
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
    newdf = pd.DataFrame(newlists, columns=[f'{target_col}_scene', f'{target_col}_driver', f'{target_col}_emotion', f'{target_col}_background'])
    return newdf

if st.button("全て実行する", key='all'):
    prompt_text = st.text_area(
    "４要素に要約させるためのプロンプトを入力してください",
    '''あなたは優秀なアシスタントです。
以下のアンケートへの回答において、回答者についてシーン（場面）・ドライバー（厳選要因）・エモーション（情緒）・バックグラウンド（背景要因）を整理してそれぞれ回答してください。
それぞれ必ず改行で区切って、それぞれ文章で記述してください。''') 
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
        savefile(df, "４要素要約済みデータ")
# 2
    progress_count = 0
    progress_total = 0
    prompt_text = st.text_area(
    "エモーションの要約を元に深掘りするためのプロンプトを入力してください",
    '''あなたは優秀なアシスタントです。
以下のアンケートの回答において、回答者が{emotion}と感じている理由と裏にある価値観を教えてください。''') 
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)

    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="emotionDeepdive", prompt_text=prompt_text, using_col=["emotion"],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("エモーション深掘り済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "エモーション深掘り済みデータ")

#3
    progress_count = 0
    progress_total = 0
    prompt_text = st.text_area(
    "価値を端的に表現するためのプロンプトを入力してください",
    '''あなたは優秀なアシスタントです。
では、この人物がこのエピソードに感じている価値を端的に表現してみて。なぜこのことが気に入っているのか、端的に表現してみて。
''') # 文字入力(複数行)

    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)

    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="value", prompt_text=prompt_text, using_col=[],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("価値を端的に表現済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "価値を端的に表現済みデータ")



#4
    progress_count = 0
    progress_total = 0
    prompt_text = st.text_area(
    "価値を深掘りするためのプロンプトを入力してください",
    '''この人が{value}を求めているのは、なぜでしょうか。 この人の価値観を端的に表現してみて。
''') 
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)

    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="value_deepdive", prompt_text=prompt_text, using_col=["value"],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("価値の深掘り済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "価値の深掘り済みデータ")

#5
    progress_count = 0
    progress_total = 0
    prompt_text = st.text_area(
    "不満・身充足を明らかにするためのプロンプトを入力してください",
    '''{value}において、{category}で解消することができる、いまの消費者の隠れた不満や未充足欲求は何ですか？
''')
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)

    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="camplain", prompt_text=prompt_text, using_col=["value","category"],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("不満・未充足を明らかにする済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "不満・未充足を明らかにする済みデータ")

#6
    progress_count = 0
    progress_total = 0
    prompt_text = st.text_area(
    "社会潮流・消費者トレンドを明らかにするためのプロンプトを入力してください",
    '''{value}が、いま求められている背景には、どんな「社会潮流」や「消費者トレンド」がありますか？３つ挙げてください
''')
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)

    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="trend", prompt_text=prompt_text, using_col=["value"],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("社会潮流・消費者トレンドを明らかにする済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "社会潮流・消費者トレンドを明らかにする済みデータ")
    
#7
    progress_count = 0
    progress_total = 0
    prompt_text = st.text_area(
    "新商品コンセプトを考えるためのプロンプトを入力してください",
    '''この人が欲しがりそうな{category}の新商品コンセプトを200文字で作成してください
''')
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)

    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="concept", prompt_text=prompt_text, using_col=["category"],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("新商品コンセプトを考える済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "新商品コンセプトを考える済みデータ")

#8
    progress_count = 0
    progress_total = 0
    prompt_text = st.text_area(
    "キャッチコピー案を考えるためのプロンプトを入力してください",
    '''あなたは、日本の著名なコピーライターです。 新商品「{value}」の広告キャッチコピーを考えてみて。
''')
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)

    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="copy", prompt_text=prompt_text, using_col=["value"],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("キャッチコピー案を考える済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "キャッチコピー案を考える済みデータ")

#9
    progress_count = 0
    progress_total = 0
    prompt_text = st.text_area(
    "販売ルートを考えるためのプロンプトを入力してください",
    '''新商品「{value}」の販売ルートを５つ考えてみてください
''')
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)

    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="route", prompt_text=prompt_text, using_col=["value"],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("販売ルートを考える済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "販売ルートを考える済みデータ")

#10
    progress_count = 0
    progress_total = 0
    prompt_text = st.text_area(
    "SoBを考えるためのプロンプトを入力してください",
    '''いま消費者が求めている欲求のひとつに、「{value}」が挙げられます。
この欲求を充たしいている商品やサービスを５つ挙げてください。
''')
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)


    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="sob", prompt_text=prompt_text, using_col=["value"],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("SoBを考える済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "SoBを考える済みデータ")
    savefinalfile(df, "")
    st.success('最終データダウンロード完了')







"""
4要素要約 スタート
"""
st.subheader('①４要素要約')
prompt_text = st.text_area(
    "４要素に要約させるためのプロンプトを入力してください",
    '''あなたは優秀なアシスタントです。
以下のアンケートへの回答において、回答者についてシーン（場面）・ドライバー（厳選要因）・エモーション（情緒）・バックグラウンド（背景要因）を整理してそれぞれ回答してください。
それぞれ必ず改行で区切って、それぞれ文章で記述してください。''') 

def find_text_start_from(keyword,text):
   search = keyword +".+"
   result = re.search(search, text)
   if result == None:
       return None
   else:
       return result.group(0).replace(keyword,"").strip()

# 1. 4要素要約
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
    newdf = pd.DataFrame(newlists, columns=[f'{target_col}_scene', f'{target_col}_driver', f'{target_col}_emotion', f'{target_col}_background'])
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
    savefile(df, "４要素要約済みデータ")

"""
4要素要約 完了
"""



# 属性情報、プロンプトテキスト、[その他の参照するカラムのリスト]を受け取って、GPTの返答テキストを返す
def ask_gpt(attributesInfo, prompt_text,human_input, addons):
    template = prompt_text+"""
回答者の属性情報: {attributesInfo}
回答: {human_input}
Assistant:
        """
    if len(addons) == 0:
        template = prompt_text+"""
        回答者の属性情報: {attributesInfo}
        回答: {human_input}
        Assistant:
        """
        prompt = PromptTemplate(
            input_variables=["attributesInfo", "human_input"],
            template=template
        )
        print(prompt)
        chatgpt_chain = LLMChain(
            llm=OpenAI(temperature=0, openai_api_key=api_key, model_name=model_name), 
            prompt=prompt, 
            verbose=True
        )
        output = chatgpt_chain.predict(attributesInfo=attributesInfo, human_input=human_input) 
        print(output)
    elif len(addons) == 1:
        pass
        for key, value in addons.items():
            exec(f"{key} = '{value}'")
        key1 = list(addons.keys())[0]
        value1 = list(addons.items())[0]
        if key1 == 'emotion':
            prompt = PromptTemplate(
                input_variables=["attributesInfo", "human_input", "emotion"],
                template=template
                )
            print(prompt)
            chatgpt_chain = LLMChain(
                llm=OpenAI(temperature=0, openai_api_key=api_key, model_name=model_name), 
                prompt=prompt, 
                verbose=True
            )
            output = chatgpt_chain.predict(attributesInfo=attributesInfo, human_input=human_input, emotion=addons['emotion'])
            print(output)
        elif key1 == "category":
            prompt = PromptTemplate(
                input_variables=["attributesInfo", "human_input", "category"],
                template=template
                )
            print(prompt)
            chatgpt_chain = LLMChain(
                llm=OpenAI(temperature=0, openai_api_key=api_key, model_name=model_name), 
                prompt=prompt, 
                verbose=True
            )
            output = chatgpt_chain.predict(attributesInfo=attributesInfo, human_input=human_input, category=addons['category'])
            print(output)
        elif key1 == "value":
            prompt = PromptTemplate(
                input_variables=["attributesInfo", "human_input", "value"],
                template=template
                )
            print(prompt)
            chatgpt_chain = LLMChain(
                llm=OpenAI(temperature=0, openai_api_key=api_key, model_name=model_name), 
                prompt=prompt, 
                verbose=True
            )
            output = chatgpt_chain.predict(attributesInfo=attributesInfo, human_input=human_input, value=addons['value'])
            print(output)
    elif len(addons) == 2:
        # 追加情報がある場合、内容を変数として定義する
        for key, value in addons.items():
            exec(f"{key} = '{value}'")
        key1 = list(addons.keys())[0]
        value1 = list(addons.items())[0]
        key2 = list(addons.keys())[1]
        value2 = list(addons.items())[1]
        if set([key1, key2]) == set(['emotion', 'value']):
            prompt = PromptTemplate(
                input_variables=["attributesInfo", "human_input", "emotion", "value"],
                template=template
                )
            print(prompt)
            chatgpt_chain = LLMChain(
                llm=OpenAI(temperature=0, openai_api_key=api_key, model_name=model_name), 
                prompt=prompt, 
                verbose=True
            )
            output = chatgpt_chain.predict(attributesInfo=attributesInfo, human_input=human_input, emotion=addons['emotion'], value=addons['value'])
            print(output)
        elif set([key1, key2]) == set(['category', 'value']):
            prompt = PromptTemplate(
                input_variables=["attributesInfo", "human_input", "category", "value"],
                template=template
                )
            print(prompt)
            chatgpt_chain = LLMChain(
                llm=OpenAI(temperature=0, openai_api_key=api_key, model_name=model_name), 
                prompt=prompt, 
                verbose=True
            )
            output = chatgpt_chain.predict(attributesInfo=attributesInfo, human_input=human_input, category=addons['category'], value=addons['value'])
            print(output)
        else:
            st.error('addons読み込みで想定外が起きました', icon="🚨")
            print('error!!!!!!!!')

    else:
        st.error('3つ以上の追加情報をGPTへの質問に利用しようとしています', icon="🚨")
        print('3つ以上')

    return output

# 対象カラム名、新しいカラム名、(属性情報)、プロンプトテキスト、利用カラムリスト（0-2こ）　を受け取って dfを返す
# using_colはリスト
def make_answer_df(target_col, new_col_name, prompt_text, using_col):
    newlists = []
    global progress_count

    if len(using_col) == 0:
        for row in zip(df[target_cols_attributes], df[target_col]):
            attributesInfo = row[0]
            human_input = row[1]
            addons = {}
            sumups = ask_gpt(attributesInfo, prompt_text, human_input, addons)
            # 実行カウント
            # global progress_count
            progress_count = progress_count + 1
            my_bar.progress(progress_count/progress_total, text=progress_text)

            newlists.append(sumups)
    elif len(using_col) == 1:
        if using_col == ['emotion']:
            for row in zip(df[target_cols_attributes], df[target_col], df[target_col+'_emotion']):
                attributesInfo = row[0]
                human_input = row[1]
                addons = {"emotion":row[2]}
                sumups = ask_gpt(attributesInfo, prompt_text, human_input, addons)
                # 実行カウント
                # global progress_count
                progress_count = progress_count + 1
                my_bar.progress(progress_count/progress_total, text=progress_text)

                newlists.append(sumups)
        elif using_col == ['category']:
            for row in zip(df[target_cols_attributes], df[target_col], df['category']):
                attributesInfo = row[0]
                human_input = row[1]
                addons = {"category":row[2]}
                sumups = ask_gpt(attributesInfo, prompt_text, human_input, addons)
                # 実行カウント
                # global progress_count
                progress_count = progress_count + 1
                my_bar.progress(progress_count/progress_total, text=progress_text)

                newlists.append(sumups)
        elif using_col == ['value']:
            for row in zip(df[target_cols_attributes], df[target_col], df[target_col+'_value']):
                attributesInfo = row[0]
                human_input = row[1]
                addons = {"value":row[2]}
                sumups = ask_gpt(attributesInfo, prompt_text, human_input, addons)
                # 実行カウント
                # global progress_count
                progress_count = progress_count + 1
                my_bar.progress(progress_count/progress_total, text=progress_text)

                newlists.append(sumups)
        else:
            print("error!!")
    elif len(using_col) == 2:
        if set(using_col) == set(['category','value']):
            for row in zip(df[target_cols_attributes], df[target_col], df['category'], df[target_col+'_value']):
                attributesInfo = row[0]
                human_input = row[1]
                addons = {"category":row[2],"value":row[3]}
                sumups = ask_gpt(attributesInfo, prompt_text, human_input, addons)
                # 実行カウント
                # global progress_count
                progress_count = progress_count + 1
                my_bar.progress(progress_count/progress_total, text=progress_text)

                newlists.append(sumups)
        elif set(using_col) == set(['emotion','value']):
            for row in zip(df[target_cols_attributes], df[target_col], df[target_col+'_emotion'], df[target_col+'_value']):
                attributesInfo = row[0]
                human_input = row[1]
                addons = {"emotion":row[2],"value":row[3]}
                sumups = ask_gpt(attributesInfo, prompt_text, human_input, addons)
                # 実行カウント
                # global progress_count
                progress_count = progress_count + 1
                my_bar.progress(progress_count/progress_total, text=progress_text)

        
    else:
        st.error('addons読み込みで想定外が起きました', icon="🚨")
        print('error!!!!!!!!')
    
    newdf = pd.DataFrame(newlists, columns=[f'{target_col}_{new_col_name}'])
    return newdf



"""
2. emotion深掘り 開始
"""
st.subheader('②エモーション深掘り')
prompt_text = st.text_area(
    "エモーションの要約を元に深掘りするためのプロンプトを入力してください",
    '''あなたは優秀なアシスタントです。
以下のアンケートの回答において、回答者が{emotion}と感じている理由と裏にある価値観を教えてください。''') 


if st.button("エモーションの要約を元に深掘り", key='emotion'):
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)

    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="emotionDeepdive", prompt_text=prompt_text, using_col=["emotion"],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("エモーション深掘り済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "エモーション深掘り済みデータ")
"""
2. emotion深掘り 完了
"""

"""
3. 価値を端的に表現 開始
"""

st.subheader('③価値を端的に表現')


prompt_text = st.text_area(
    "価値を端的に表現するためのプロンプトを入力してください",
    '''あなたは優秀なアシスタントです。
では、この人物がこのエピソードに感じている価値を端的に表現してみて。なぜこのことが気に入っているのか、端的に表現してみて。
''') # 文字入力(複数行)


if st.button("価値を端的に表現", key='value'):
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)

    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="value", prompt_text=prompt_text, using_col=[],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("価値を端的に表現済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "価値を端的に表現済みデータ")

"""
3. 価値を端的に表現 完了
"""


"""
4. 価値の深掘り 開始
"""
st.subheader('④価値の深掘り')
prompt_text = st.text_area(
    "価値を深掘りするためのプロンプトを入力してください",
    '''この人が{value}を求めているのは、なぜでしょうか。 この人の価値観を端的に表現してみて。
''') 
if st.button("価値の深掘り", key='value_deepdive'):
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)

    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="value_deepdive", prompt_text=prompt_text, using_col=["value"],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("価値の深掘り済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "価値の深掘り済みデータ")

"""
4. 価値の深掘り 完了
"""

"""
5. 不満・未充足 開始
"""
st.subheader('⑤不満・未充足')
prompt_text = st.text_area(
    "不満・身充足を明らかにするためのプロンプトを入力してください",
    '''{value}において、{category}で解消することができる、いまの消費者の隠れた不満や未充足欲求は何ですか？
''')
if st.button("不満・未充足を明らかにする", key='complain'):
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)

    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="camplain", prompt_text=prompt_text, using_col=["value","category"],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("不満・未充足を明らかにする済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "不満・未充足を明らかにする済みデータ")

"""
5. 不満・未充足 完了
"""


# 6. 社会潮流・消費者トレンド
st.subheader('⑥社会潮流・消費者トレンド')
prompt_text = st.text_area(
    "社会潮流・消費者トレンドを明らかにするためのプロンプトを入力してください",
    '''{value}が、いま求められている背景には、どんな「社会潮流」や「消費者トレンド」がありますか？３つ挙げてください
''')
if st.button("社会潮流・消費者トレンドを明らかにする", key='trend'):
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)

    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="trend", prompt_text=prompt_text, using_col=["value"],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("社会潮流・消費者トレンドを明らかにする済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "社会潮流・消費者トレンドを明らかにする済みデータ")
    
# 7. 新商品コンセプト
st.subheader('⑦新商品コンセプト')
prompt_text = st.text_area(
    "新商品コンセプトを考えるためのプロンプトを入力してください",
    '''この人が欲しがりそうな{category}の新商品コンセプトを200文字で作成してください
''')
if st.button("新商品コンセプトを考える", key='concept'):
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)

    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="concept", prompt_text=prompt_text, using_col=["category"],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("新商品コンセプトを考える済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "新商品コンセプトを考える済みデータ")

# 8. キャッチコピー案
st.subheader('⑧キャッチコピー案')
prompt_text = st.text_area(
    "キャッチコピー案を考えるためのプロンプトを入力してください",
    '''あなたは、日本の著名なコピーライターです。 新商品「{value}」の広告キャッチコピーを考えてみて。
''')
if st.button("キャッチコピー案を考える", key='copy'):
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)

    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="copy", prompt_text=prompt_text, using_col=["value"],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("キャッチコピー案を考える済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "キャッチコピー案を考える済みデータ")

# 9. 販売ルート
st.subheader('⑨販売ルート')
prompt_text = st.text_area(
    "販売ルートを考えるためのプロンプトを入力してください",
    '''新商品「{value}」の販売ルートを５つ考えてみてください
''')
if st.button("販売ルートを考える", key='route'):
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)

    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="route", prompt_text=prompt_text, using_col=["value"],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("販売ルートを考える済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "販売ルートを考える済みデータ")

# 10. SoB
st.subheader('⑩SoB')
prompt_text = st.text_area(
    "SoBを考えるためのプロンプトを入力してください",
    '''いま消費者が求めている欲求のひとつに、「{value}」が挙げられます。
この欲求を充たしいている商品やサービスを５つ挙げてください。
''')
if st.button("SoBを考える", key='sob'):
    test_apikey()
    df = st.session_state['df']
    # プログレスバー
    progress_text = "実行中...処理が終わるまでしばらくお待ちください。"
    progress_total = len(target_cols) * df.shape[0]
    my_bar = st.progress(0, text=progress_text)


    # 全てのカラムに対して処理
    for target_col in target_cols:
        newdf = make_answer_df(target_col, new_col_name="sob", prompt_text=prompt_text, using_col=["value"],)
        st.info(f'{target_col}について処理完了')
        df = pd.concat([df, newdf], axis=1) 
    st.success('要約完了')
    st.text("SoBを考える済みデータ")
    st.write(df)
    st.session_state['df'] = df
    output_csv = df.to_csv('output.csv', encoding="utf-8")
    savefile(df, "SoBを考える済みデータ")

if st.button("最終ファイルをダウンロード", key='final_download'):
    df = st.session_state['df']
    savefinalfile(df, "")
# st.button("全て実行する", key='all')
