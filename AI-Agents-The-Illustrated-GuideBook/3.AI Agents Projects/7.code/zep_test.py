import os, uuid
from xmlrpc import client

from zep_cloud.client import Zep

API_KEY = "sdfsdffsf.sdfsdf-sdfsdf"

zep_client = Zep(api_key=API_KEY)

# You can choose any user ID, but we recommend using your internal user ID
user_id = "zep_sandbox_79df05c4e259_6"
new_user = zep_client.user.add(
    user_id=user_id,
    email="6jane.smith@example.com",
    first_name="6Jane",
    last_name="Smith",
)

thread_id = uuid.uuid4().hex # A new thread identifier
zep_client.thread.create(
    thread_id=thread_id,
    user_id=user_id,
)

from zep_cloud.types import Message
from datetime import datetime, timezone

messages = [
    Message(
        created_at=datetime.now(timezone.utc).isoformat(),
        name="Jane Smith6",
        role="user",
        content="Who was Octavia Butler?",
    )
]
response = zep_client.thread.add_messages(thread_id, messages=messages)

print(response)


import json

# Example: User listened to a song in your application
event_data = {
    "user_id": user_id,
    "user_name": "Jane Smith6",
    "event_type": "song_played",
    "song_title": "Bohemian Rhapsody",
    "artist": "Queen",
    "duration_seconds": 354
}
zep_client.graph.add(
    user_id=user_id,
    type="json",
    data=json.dumps(event_data)
)


# Get context for the thread
user_context = zep_client.thread.get_user_context(thread_id=thread_id)
# Access the context block for use as untrusted model input
context_block = user_context.context
print(context_block)