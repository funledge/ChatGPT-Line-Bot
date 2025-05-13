from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, AudioMessage
)
import os
import uuid
import requests
from datetime import datetime

from src.models import OpenAIModel
from src.memory import Memory
from src.logger import logger
from src.storage import Storage, FileStorage, MongoStorage
from src.utils import get_role_and_content
from src.service.youtube import Youtube, YoutubeTranscriptReader
from src.service.website import Website, WebsiteReader
from src.mongodb import mongodb

load_dotenv('.env')

GAS_URL = "https://script.google.com/macros/s/AKfycbwwYbSuxJE0N2ExDu-gHuRH7TDIhB92jKZydr-uQ-WW9L2PTFjNA3ZP6Y7HBYhXHxA/exec"

MAX_DAILY_LIMIT = 5

def update_usage(user_id):
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        requests.post(GAS_URL, json={
            "user_id": user_id,
            "date": today,
            "count": 1
        }, timeout=2)
        return True
    except Exception as e:
        print("スプレッドシート更新失敗:", e)
        return False

def is_over_limit(user_id):
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        res = requests.post(GAS_URL, json={
            "user_id": user_id,
            "date": today,
            "check_only": True
        }, timeout=2)
        return res.json().get("count", 0) >= MAX_DAILY_LIMIT
    except Exception as e:
        print("チェック失敗", e)
        return False

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))
storage = None
youtube = Youtube(step=4)
website = Website()

memory = Memory(system_message=os.getenv('SYSTEM_MESSAGE'), memory_message_count=2)
model_management = {}
api_keys = {}

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()
    print("✅ メッセージ受信:", user_id, text)  # ←これ追加！
    logger.info(f'{user_id}: {text}')

    try:
        if text.startswith('/註冊'):
            api_key = text[3:].strip()
            model = OpenAIModel(api_key=api_key)
            is_successful, _, _ = model.check_token_valid()
            if not is_successful:
                raise ValueError('Invalid API token')
            model_management[user_id] = model
            storage.save({user_id: api_key})
            msg = TextSendMessage(text='Token 有效，註冊成功')

        elif text.startswith('/指令說明'):
            msg = TextSendMessage(text="指令：\n/註冊 + API Token\n👉 API Token 請先到 https://platform.openai.com/ 註冊登入後取得\n\n/系統訊息 + Prompt\n👉 Prompt 可以命令機器人扮演某個角色，例如：請你扮演擅長做總結的人\n\n/清除\n👉 當前每一次都會紀錄最後兩筆歷史紀錄，這個指令能夠清除歷史訊息\n\n/圖像 + Prompt\n👉 會調用 DALL∙E 2 Model，以文字生成圖像\n\n語音輸入\n👉 會調用 Whisper 模型，先將語音轉換成文字，再調用 ChatGPT 以文字回覆\n\n其他文字輸入\n👉 調用 ChatGPT 以文字回覆")

        elif text.startswith('/系統訊息'):
            memory.change_system_message(user_id, text[5:].strip())
            msg = TextSendMessage(text='輸入成功')

        elif text.startswith('/清除'):
            memory.remove(user_id)
            msg = TextSendMessage(text='歷史訊息清除成功')

        elif text.startswith('/圖像'):
            prompt = text[3:].strip()
            memory.append(user_id, 'user', prompt)
            is_successful, response, error_message = model_management[user_id].image_generations(prompt)
            if not is_successful:
                raise Exception(error_message)
            url = response['data'][0]['url']
            msg = ImageSendMessage(
                original_content_url=url,
                preview_image_url=url
            )
            memory.append(user_id, 'assistant', url)

        else:
            if is_over_limit(user_id):
                msg = TextSendMessage(text='今日の無料利用回数（5回）を超えました！続けて利用したい場合は有料プランをご検討ください😊')
            else:
                user_model = model_management.get(user_id)
                if not user_model:
                    user_model = OpenAIModel(api_key=os.getenv('OPENAI_API_KEY'))
                    model_management[user_id] = user_model

                prompt = f"""
あなたは英語添削をする先生です。
以下の英文を添削してください。

・まず、文章の良い点や間違いを指摘（英語＆日本語）
・次に、添削後の正しい英文を示す
・最後に、日本語で初心者向けの簡単なアドバイスを添える

フォーマットは以下の通りです：

【添削結果】
（英語のコメント）
（日本語のコメント）
→ 添削後の正しい英文

【アドバイス】
（日本語で一言アドバイス）

対象の英文：
「{text}」
"""
                is_successful, response, error_message = user_model.chat_completions([
                    {'role': 'user', 'content': prompt}
                ], os.getenv('OPENAI_MODEL_ENGINE'))
                if not is_successful:
                    raise Exception(error_message)
                msg = TextSendMessage(text=response)
                try:
                    update_usage(user_id)
                except Exception as e:
                    logger.error(f"update_usage failed: {e}")

    except ValueError:
        msg = TextSendMessage(text='Token 無效，請重新註冊，格式為 /註冊 sk-xxxxx')
    except KeyError:
        msg = TextSendMessage(text='請先註冊 Token，格式為 /註冊 sk-xxxxx')
    except Exception as e:
        memory.remove(user_id)
        if str(e).startswith('Incorrect API key provided'):
            msg = TextSendMessage(text='OpenAI API Token 有誤，請重新註冊。')
        elif str(e).startswith('That model is currently overloaded with other requests.'):
            msg = TextSendMessage(text='已超過負荷，請稍後再試')
        else:
            msg = TextSendMessage(text=str(e))
    line_bot_api.reply_message(event.reply_token, msg)

@app.route("/", methods=['GET'])
def home():
    return 'Hello World'

if __name__ == "__main__":
    if os.getenv('USE_MONGO'):
        mongodb.connect_to_database()
        storage = Storage(MongoStorage(mongodb.db))
    else:
        storage = Storage(FileStorage('db.json'))

    try:
        data = storage.load()
        for user_id in data.keys():
            model_management[user_id] = OpenAIModel(api_key=data[user_id])
    except FileNotFoundError:
        pass

    app.run(host="0.0.0.0", port=8080, debug=False)
