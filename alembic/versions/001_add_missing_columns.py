"""add missing columns

Revision ID: 001
Create Date: 2025-05-30
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Add confidence column to chat_messages if it doesn't exist
    try:
        op.add_column('chat_messages', sa.Column('confidence', sa.Float(), nullable=True))
    except:
        pass  # Column might already exist

def downgrade():
    op.drop_column('chat_messages', 'confidence') 