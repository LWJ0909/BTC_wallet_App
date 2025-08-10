# 💰 BTC Wallet App

## 📌 프로젝트 개요
**BTC Wallet App**은 비트코인 공개키(XPUB) 또는 주소로 지갑 **잔액**과 **매입원가 대비 수익률(DCA 기준)** 을 계산해 주는 읽기 전용(Watch-only) 웹 앱입니다.  
데이터는 **mempool.space(Esplora)** 와 **Binance / Upbit** 공개 API를 사용합니다.  
*개인키/시드는 절대 요구하지 않습니다.*

## ✨ 주요 기능
- **잔액 조회(UTXO 기반)**: 현재 보유 BTC 합계를 빠르게 계산
- **원가/미실현 손익**: 각 입금 날짜의 종가(UTC)로 원가 산정 후 현재가와 비교
- **실현 손익(옵션)**: 출금 시 FIFO(선입선출)로 원가 차감하여 실현 손익 집계
- **XPUB/Descriptor 스캔**: 외부(0/*), 변경(1/*) 체인 파생 및 갭리밋 스캔
- **시각화**: 누적 원가 vs 평가액 그래프 제공

## 🛠 기술 스택
- Python 3.11+
- Streamlit (UI)
- bip-utils (XPUB 파생)
- Plotly (그래프)
- APIs: mempool.space, Binance, Upbit

## 📦 설치
```bash
git clone https://github.com/YOUR_ID/BTC_wallet_App.git
cd BTC_wallet_App

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 🚀 실행
```bash
streamlit run app.py
```

## 🔑 사용 방법
주소 모드: BTC 주소들을 줄바꿈으로 입력 → 잔액/미실현 손익 계산

XPUB 모드: zpub/ypub/xpub 또는 Descriptor 입력 → 파생주소 스캔 후 전체 계산
(에어갭 사용자는 watch-only 공개키만 사용하세요.)

## ⚠️ 보안/한계
이 앱은 읽기 전용이며 개인키/시드를 절대 요구하지 않습니다.

퍼블릭 API의 요청 제한과 네트워크 상태에 따라 스캔 속도가 달라질 수 있습니다.

가격은 일봉 종가(UTC) 를 사용하며 실제 체결가와 다를 수 있습니다.

투자 판단의 책임은 사용자 본인에게 있습니다.

## 📜 라이선스
MIT License
