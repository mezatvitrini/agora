@echo off
cd /d D:\PycharmProjects\agora
call .venv\Scripts\activate.bat
python -m streamlit run app.py
pause
