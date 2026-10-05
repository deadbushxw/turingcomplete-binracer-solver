@echo off
chcp 65001 >nul
rem 启动后控制台会自动最小化到任务栏(不遮挡游戏), 鼠标悬停任务栏图标可看答题进度
rem 想看实时输出: 从任务栏恢复该窗口即可; 按 F12 中止
cd /d "%~dp0"
if not exist logs mkdir logs
start "二进制速算机器人" python binracer\bot.py --input=sendinput --capture=screen --max-seconds 1800 --save-shots --log-file "logs\standalone_stdout.txt"
exit
