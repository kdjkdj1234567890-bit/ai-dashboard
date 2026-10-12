"""AI 데이터 대시보드 — 시연용 프로토타입 (Flask)

CSV 업로드 → 컬럼 자동 인식 → 차트 자동 생성 → 룰 기반 인사이트 자동 생성.
외부 API 호출 없음. 로컬에서 완전히 동작.
모든 예외는 사용자에게 친절한 안내 메시지로 변환되며, traceback은 노출하지 않음.
"""
import io
import logging
import os
import traceback
import uuid

import pandas as pd
from flask import Flask, redirect, render_template, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from insights import analyze, detect_columns

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV = os.path.join(BASE, "sample_data", "monthly_sales_2025.csv")

DATASTORE: dict[str, dict] = {}  # token -> {"df": DataFrame, "filename": str}

ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

TYPE_LABEL = {"date": "날짜", "numeric": "숫자", "category": "범주"}
LEVEL_ICON = {"up": "📈", "down": "📉", "warning": "⚠️", "info": "💡"}

log = logging.getLogger("aidash")


def read_csv_smart(file_bytes: bytes) -> pd.DataFrame:
    for enc in ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # 마지막 수단: 오류 무시하고 디코딩
    text = file_bytes.decode("utf-8", errors="ignore")
    return pd.read_csv(io.StringIO(text))


def validate_df(df: pd.DataFrame) -> str | None:
    """문제 있으면 안내 메시지, 없으면 None."""
    if df.empty:
        return "파일에 데이터 행이 없습니다. 내용이 있는 CSV를 올려주세요."
    if len(df.columns) < 2:
        return "분석할 컬럼이 부족합니다. 2개 이상의 컬럼이 필요합니다."
    coltypes = detect_columns(df)
    if not any(t == "numeric" for t in coltypes.values()):
        return ("숫자형 컬럼을 찾지 못했습니다. 매출·수량·금액 등 숫자 데이터가 "
                "포함된 CSV를 올려주세요.")
    return None


def store_df(df: pd.DataFrame, filename: str) -> str:
    token = uuid.uuid4().hex[:12]
    DATASTORE[token] = {"df": df, "filename": filename}
    # 메모리 보호: 최대 50개까지만 보관
    while len(DATASTORE) > 50:
        DATASTORE.pop(next(iter(DATASTORE)))
    return token


@app.errorhandler(RequestEntityTooLarge)
def too_large(_e):
    return render_template("index.html",
                           error="파일이 너무 큽니다. 10MB 이하의 CSV를 올려주세요."), 413


@app.errorhandler(500)
def internal_error(_e):
    return render_template(
        "index.html",
        error="분석 중 오류가 발생했습니다. 파일 형식을 확인하고 다시 시도해주세요."), 500


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return render_template("index.html", error="CSV 파일을 선택해주세요.")
    if not f.filename.lower().endswith(".csv"):
        return render_template("index.html",
                               error="CSV 파일만 업로드할 수 있습니다. (.csv 확장자 필요)")
    try:
        raw = f.read()
    except Exception:
        return render_template("index.html", error="파일을 읽지 못했습니다. 다시 시도해주세요.")
    if not raw.strip():
        return render_template("index.html", error="빈 파일입니다. 내용이 있는 CSV를 올려주세요.")
    try:
        df = read_csv_smart(raw)
    except Exception:
        log.warning("CSV parse failed:\n%s", traceback.format_exc())
        return render_template("index.html",
                               error="CSV 형식을 해석하지 못했습니다. 쉼표(,)로 구분된 표준 CSV인지 확인해주세요.")
    problem = validate_df(df)
    if problem:
        return render_template("index.html", error=problem)
    token = store_df(df, f.filename)
    return redirect(url_for("dashboard", t=token))


@app.route("/sample")
def sample():
    try:
        df = read_csv_smart(open(SAMPLE_CSV, "rb").read())
    except Exception:
        log.warning("Sample load failed:\n%s", traceback.format_exc())
        return render_template("index.html", error="샘플 데이터를 불러오지 못했습니다.")
    token = store_df(df, "monthly_sales_2025.csv (샘플)")
    return redirect(url_for("dashboard", t=token))


@app.route("/sample.csv")
def sample_download():
    return send_file(SAMPLE_CSV, as_attachment=True,
                     download_name="monthly_sales_2025.csv", mimetype="text/csv")


@app.route("/dashboard")
def dashboard():
    token = request.args.get("t", "")
    item = DATASTORE.get(token)
    if not item:
        return redirect(url_for("index"))
    df = item["df"]
    try:
        result = analyze(df)
    except Exception:
        log.warning("Analyze failed:\n%s", traceback.format_exc())
        return render_template(
            "index.html",
            error="데이터를 분석하지 못했습니다. 컬럼 구조를 확인하고 다시 시도해주세요.")
    insights = result["insights"]
    featured = insights[0] if insights else None
    rest = insights[1:] if len(insights) > 1 else []
    has_date = any(c["type"] == "date" for c in result["columns"])
    return render_template(
        "dashboard.html",
        filename=item["filename"],
        n_rows=result["n_rows"],
        columns=result["columns"],
        type_label=TYPE_LABEL,
        level_icon=LEVEL_ICON,
        has_date=has_date,
        kpis=result["kpis"],
        charts=result["charts"],
        featured=featured,
        insights=rest,
    )


if __name__ == "__main__":
    print("AI 데이터 대시보드 (시연용 프로토타입)")
    print("브라우저에서 http://127.0.0.1:5000 접속")
    app.run(host="127.0.0.1", port=5000, debug=False)
