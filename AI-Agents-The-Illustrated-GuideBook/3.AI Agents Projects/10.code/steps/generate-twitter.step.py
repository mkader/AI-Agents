import json
from pydantic import BaseModel, HttpUrl
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from ai_provider import create_ai_client

load_dotenv()

class GenerateInput(BaseModel):
    requestId: str
    url: HttpUrl
    title: str
    content: str
    timestamp: int

config = {
    'type': 'event',
    'name': 'TwitterGenerate',
    'description': 'Generates Twitter content',
    'subscribes': ['generate-content'],
    'emits': ['twitter-schedule'],
    'input': GenerateInput.model_json_schema(),
    'flows': ['content-generation']
}

async def handler(input, context):
    try:
        openai_client, model = create_ai_client()
        with open(Path(__file__).resolve().parent.parent / "twitter-prompt.txt", "r", encoding='utf-8') as f:
            twitterPromptTemplate = f.read()
        
        twitterPrompt = twitterPromptTemplate.replace('{{title}}', input['title']).replace('{{content}}', input['content'])

        context.logger.info("🔄 Twitter content generation started...")

        twitter_content = await openai_client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': twitterPrompt}],
                temperature=0.7,
                max_tokens=2000,
                response_format={'type': 'json_object'}
        )  

        message_content = twitter_content.choices[0].message.content
        if not message_content:
            raise ValueError(f'{provider} returned no Twitter content')
        twitter_content = json.loads(message_content)

        context.logger.info(f"🎉 Twitter content generated successfully!")

        await context.emit({
            'topic': 'twitter-schedule',
            'data': {
                'requestId': input['requestId'],
                'url': input['url'],
                'title': input['title'],
                'content': twitter_content,
                'generatedAt': datetime.now().isoformat(),
                'originalUrl': input['url']
            }
        })
    except Exception as e:
        context.logger.error(f"❌ Content generation failed: {e}")
        raise e