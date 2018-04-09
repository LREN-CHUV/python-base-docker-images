#!/usr/bin/env python3

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, DateTime
from sqlalchemy.sql import func


Base = declarative_base()


class JobResult(Base):
    __tablename__ = 'job_result'

    job_id = Column(String(128), primary_key=True)
    node = Column(String(32), primary_key=True)
    timestamp = Column(DateTime, server_default=func.now())
    data = Column(String)
    error = Column(String(256))
    shape = Column(String(256))
    function = Column(String(256))
