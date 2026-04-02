from __future__ import annotations

import logging

from app.core.database import Base, SessionLocal, engine
from app.models.user import User, UserRole
from app.services.storage import ensure_dirs
from app.services.users import create_user

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    ensure_dirs()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        if not db.query(User).filter(User.username == "admin").first():
            create_user(db, "admin", "admin123", UserRole.ADMIN)
            logger.info("Admin created.")

        if not db.query(User).filter(User.username == "user").first():
            create_user(db, "user", "user123", UserRole.USER)
            logger.info("User created.")

        logger.info("DB initialized. Users: admin/admin123, user/user123")
    except Exception as e:
        db.rollback()
        logger.error(f"Seed failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
