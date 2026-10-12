#!/bin/bash
# AI 데이터 대시보드 (시연용 프로토타입) 실행 스크립트
set -e
cd "$(dirname "$0")"

if ! python3 -c "import flask, pandas, numpy" 2>/dev/null; then
  echo "필요 패키지 설치 중..."
  pip install -r requirements.txt
fi

echo "=============================================="
echo " AI 데이터 대시보드 (시연용 프로토타입)"
echo " 브라우저에서 http://127.0.0.1:5000 접속"
echo "=============================================="
python3 app.py
