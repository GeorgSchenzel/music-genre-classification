#!/bin/sh
gunicorn --chdir mgclass --bind 0.0.0.0:8000 "server:create_app()"