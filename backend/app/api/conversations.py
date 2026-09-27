"""
Conversation history endpoint (Phase 8): a read-only way to fetch a
conversation's stored turns — for a frontend that wants to render a chat
transcript (Phase 9), or simply to inspect what a conversation_id resolved
to for debugging. Writing happens only as a side effect of `/api/chat`
(`app/api/chat.py`); there is no POST here, deliberately, since a
conversation's existence and content are governed entirely by the chat
turns that created them.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.models import Conversation
from app.database.session import get_db
from app.schemas.schemas import ConversationOut

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: str, db: Session = Depends(get_db)) -> ConversationOut:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return ConversationOut.model_validate(conversation)
