"""add media files table

Revision ID: 004
Revises: 003
Create Date: 2024-11-13

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create media_files table."""
    op.create_table(
        'media_files',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('file_url', sa.String(length=500), nullable=False),
        sa.Column('thumbnail_url', sa.String(length=500), nullable=True),
        sa.Column('file_size', sa.Integer(), nullable=False),
        sa.Column('content_type', sa.String(length=100), nullable=False),
        sa.Column('file_hash', sa.String(length=64), nullable=False),
        sa.Column('media_type', sa.Enum('video', 'image', 'audio', name='mediatype'), nullable=False),
        sa.Column('duration', sa.Integer(), nullable=True),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('fps', sa.Integer(), nullable=True),
        sa.Column('processing_status', sa.Enum('pending', 'processing', 'completed', 'failed', name='processingstatus'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_public', sa.Boolean(), nullable=True),
        sa.Column('uploaded_at', sa.DateTime(), nullable=False),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_media_files_file_hash'), 'media_files', ['file_hash'], unique=False)


def downgrade() -> None:
    """Drop media_files table."""
    op.drop_index(op.f('ix_media_files_file_hash'), table_name='media_files')
    op.drop_table('media_files')
    op.execute('DROP TYPE mediatype')
    op.execute('DROP TYPE processingstatus')
