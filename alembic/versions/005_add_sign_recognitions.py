"""add sign recognitions table

Revision ID: 005
Revises: 004
Create Date: 2024-11-13

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create sign_recognitions table."""
    op.create_table(
        'sign_recognitions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('recognized_sign_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('gsl_signs.id'), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('processing_time', sa.Float(), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('media_type', sa.String(20), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    
    # Create indexes for common queries
    op.create_index('ix_sign_recognitions_user_id', 'sign_recognitions', ['user_id'])
    op.create_index('ix_sign_recognitions_status', 'sign_recognitions', ['status'])
    op.create_index('ix_sign_recognitions_created_at', 'sign_recognitions', ['created_at'])


def downgrade() -> None:
    """Drop sign_recognitions table."""
    op.drop_index('ix_sign_recognitions_created_at', table_name='sign_recognitions')
    op.drop_index('ix_sign_recognitions_status', table_name='sign_recognitions')
    op.drop_index('ix_sign_recognitions_user_id', table_name='sign_recognitions')
    op.drop_table('sign_recognitions')
