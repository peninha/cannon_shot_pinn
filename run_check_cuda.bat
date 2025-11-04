@echo off
set PATH=%PATH%;C:\Users\millo\anaconda3\Scripts
call activate.bat PINN
python d:\Code\cannon_shot_pinn\check_cuda.py
pause

