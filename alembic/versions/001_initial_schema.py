"""Initial database schema

Revision ID: 001
Revises: 
Create Date: 2024-11-12 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create sign_categories table
    op.create_table('sign_categories',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('parent_category_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['parent_category_id'], ['sign_categories.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )

    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=100), nullable=True),
        sa.Column('age_group', sa.Enum('child', 'teen', 'adult', 'senior', name='agegroup'), nullable=True),
        sa.Column('learning_level', sa.Enum('beginner', 'intermediate', 'advanced', name='learninglevel'), nullable=True),
        sa.Column('preferred_language', sa.String(length=50), nullable=True),
        sa.Column('accessibility_needs', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('is_verified', sa.Boolean(), nullable=True),
        sa.Column('role', sa.Enum('learner', 'teacher', 'admin', name='userrole'), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)

    # Create gsl_signs table
    op.create_table('gsl_signs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('sign_name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('category_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('difficulty_level', sa.Integer(), nullable=True),
        sa.Column('video_url', sa.String(length=500), nullable=True),
        sa.Column('thumbnail_url', sa.String(length=500), nullable=True),
        sa.Column('usage_examples', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('related_signs', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('extra_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['category_id'], ['sign_categories.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_gsl_signs_sign_name'), 'gsl_signs', ['sign_name'], unique=False)

    # Create lessons table
    op.create_table('lessons',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('level', sa.Integer(), nullable=True),
        sa.Column('category', sa.String(length=100), nullable=False),
        sa.Column('sequence_order', sa.Integer(), nullable=True),
        sa.Column('signs_covered', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('learning_objectives', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('estimated_duration', sa.Integer(), nullable=True),
        sa.Column('prerequisites', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('extra_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create achievements table
    op.create_table('achievements',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('type', sa.String(length=50), nullable=False),
        sa.Column('points', sa.Integer(), nullable=True),
        sa.Column('criteria', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('icon_url', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create learning_progress table
    op.create_table('learning_progress',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('total_lessons_completed', sa.Integer(), nullable=True),
        sa.Column('current_level', sa.Integer(), nullable=True),
        sa.Column('experience_points', sa.Integer(), nullable=True),
        sa.Column('signs_learned', sa.Integer(), nullable=True),
        sa.Column('accuracy_rate', sa.Float(), nullable=True),
        sa.Column('practice_time_minutes', sa.Integer(), nullable=True),
        sa.Column('current_streak', sa.Integer(), nullable=True),
        sa.Column('longest_streak', sa.Integer(), nullable=True),
        sa.Column('last_activity', sa.DateTime(), nullable=True),
        sa.Column('days_active', sa.Integer(), nullable=True),
        sa.Column('achievements', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('completed_lessons', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('extra_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )

    # Create practice_sessions table
    op.create_table('practice_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_type', sa.String(length=50), nullable=False),
        sa.Column('lesson_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('signs_practiced', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=False),
        sa.Column('accuracy_score', sa.Float(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('extra_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['lesson_id'], ['lessons.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('practice_sessions')
    op.drop_table('learning_progress')
    op.drop_table('achievements')
    op.drop_table('lessons')
    op.drop_index(op.f('ix_gsl_signs_sign_name'), table_name='gsl_signs')
    op.drop_table('gsl_signs')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
    op.drop_table('sign_categories')