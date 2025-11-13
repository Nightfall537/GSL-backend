"""add translations

Revision ID: 006
Revises: 005
Create Date: 2024-11-13

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade():
    # Create translations table
    op.create_table(
        'translations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('source_type', sa.String(50), nullable=False),
        sa.Column('target_type', sa.String(50), nullable=False),
        sa.Column('source_content', sa.Text, nullable=True),
        sa.Column('target_content', sa.Text, nullable=True),
        sa.Column('confidence_score', sa.Float, default=0.0),
        sa.Column('processing_time', sa.Float, nullable=True),
        sa.Column('metadata', postgresql.JSONB, default={}),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )
    
    # Create indexes
    op.create_index('ix_translations_user_id', 'translations', ['user_id'])
    op.create_index('ix_translations_source_type', 'translations', ['source_type'])
    op.create_index('ix_translations_created_at', 'translations', ['created_at'])
    
    # Create sign_sequences table
    op.create_table(
        'sign_sequences',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('translation_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('translations.id'), nullable=True),
        sa.Column('original_text', sa.Text, nullable=False),
        sa.Column('sign_ids', postgresql.JSONB, nullable=False),
        sa.Column('transitions', postgresql.JSONB, default=[]),
        sa.Column('estimated_duration', sa.Float, nullable=True),
        sa.Column('grammar_applied', sa.String(50), default='harmonized_gsl'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    
    # Create indexes
    op.create_index('ix_sign_sequences_translation_id', 'sign_sequences', ['translation_id'])


def downgrade():
    op.drop_index('ix_sign_sequences_translation_id')
    op.drop_table('sign_sequences')
    
    op.drop_index('ix_translations_created_at')
    op.drop_index('ix_translations_source_type')
    op.drop_index('ix_translations_user_id')
    op.drop_table('translations')
