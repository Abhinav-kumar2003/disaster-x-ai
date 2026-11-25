"""initial schema — regions, model_versions, observations, predictions, alerts

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-16

Mirrors db/schema.sql. PostGIS DDL (geometry columns, GiST indexes) is applied
via raw SQL only on PostgreSQL; on other dialects the geometry columns are
omitted (schema.sql remains the canonical full-Postgres definition).
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy import CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_pg = bind.dialect.name == "postgresql"
    json_t = JSONB() if is_pg else sa.JSON()

    op.create_table(
        "regions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("code", sa.Text, nullable=False, unique=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("country", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("artifact_path", sa.Text, nullable=False),
        sa.Column("task", sa.Text, nullable=False),
        sa.Column("metrics", json_t, nullable=False),
        sa.Column("trained_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.UniqueConstraint("name", "artifact_path"),
    )
    op.create_table(
        "observations",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("region_id", sa.Integer,
                  sa.ForeignKey("regions.id"), nullable=False),
        sa.Column("observed_at", sa.Date, nullable=False),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("measurement", json_t, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("obs_region_time_idx", "observations",
                    ["region_id", sa.text("observed_at DESC")])
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("model_version_id", sa.Integer,
                  sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("region_id", sa.Integer, sa.ForeignKey("regions.id"),
                  nullable=False),
        sa.Column("risk_p", sa.Float, nullable=False),
        sa.Column("risk_tier", sa.Text, nullable=False),
        sa.Column("features", json_t),
        sa.Column("drivers", json_t),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.CheckConstraint("risk_p BETWEEN 0 AND 1",
                           name="pred_risk_p_check"),
    )
    op.create_table(
        "alerts",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("prediction_id", sa.BigInteger,
                  sa.ForeignKey("predictions.id"), nullable=False),
        sa.Column("region_id", sa.Integer, sa.ForeignKey("regions.id"),
                  nullable=False),
        sa.Column("severity", sa.Text, nullable=False),
        sa.Column("rule", sa.Text, nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("drivers", json_t),
        sa.Column("status", sa.Text, nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "severity IN ('watch','warning','critical')",
            name="alerts_severity_check"),
    )
    op.create_table(
        "app_users",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("email", sa.Text, nullable=False, unique=True),
        sa.Column("full_name", sa.Text),
        sa.Column("role", sa.Text, nullable=False, server_default="analyst"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.CheckConstraint("role IN ('analyst','emergency_manager','admin')",
                           name="users_role_check"),
    )
    op.create_table(
        "alert_acknowledgements",
        sa.Column("alert_id", sa.BigInteger, sa.ForeignKey("alerts.id"),
                  nullable=False),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("app_users.id"),
                  nullable=False),
        sa.Column("note", sa.Text),
        sa.Column("ack_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("alert_id", "user_id"),
    )
    op.create_index("alerts_status_idx", "alerts", ["status", "created_at"])
    op.create_index("alerts_region_idx", "alerts", ["region_id", "created_at"])
    op.create_index("pred_tier_idx", "predictions", ["risk_tier"])

    if is_pg:
        # PostGIS columns + GiST indexes (raw SQL — portable types don't
        # cover geometry)
        op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
        op.execute("ALTER TABLE regions ADD COLUMN "
                   "geometry geometry(Polygon, 4326) NOT NULL")
        op.execute("ALTER TABLE observations ADD COLUMN "
                   "geom geometry(Point, 4326) NOT NULL")
        op.execute("ALTER TABLE predictions ADD COLUMN "
                   "cell_geom geometry(Polygon, 4326) NOT NULL")
        op.execute("ALTER TABLE predictions ADD COLUMN "
                   "cell_center geometry(Point, 4326) NOT NULL")
        op.execute("CREATE INDEX obs_geom_idx ON observations "
                   "USING GIST (geom)")
        op.execute("CREATE INDEX pred_geom_idx ON predictions "
                   "USING GIST (cell_geom)")

    # seed model versions (idempotent)
    op.execute(
        "INSERT INTO model_versions (name, artifact_path, task, metrics) "
        "SELECT 'susceptibility_rf_v2', "
        "'ml/models/flood_susceptibility_v2_rf.joblib', "
        "'long-term flood susceptibility 2020-2024 from 2014-2019 evidence', "
        "'{\"test_auc\": 0.832, \"pr_auc\": 0.966, \"f1\": 0.776}' "
        "WHERE NOT EXISTS (SELECT 1 FROM model_versions "
        "WHERE name = 'susceptibility_rf_v2')"
    )


def downgrade() -> None:
    for tbl in ("alert_acknowledgements", "app_users", "alerts",
                "predictions", "observations", "model_versions", "regions"):
        op.drop_table(tbl)
