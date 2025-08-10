import streamlit as st
import pandas as pd
import requests, time, functools, datetime as dt, os
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import plotly.express as px

try:
    from bip_utils import Bip44, Bip49, Bip84, Bip86, Bip44Changes, Bip44Coins, Bip49Coins, Bip84Coins, Bip86Coins
    BIP_OK = True
except:
    BIP_OK = False

st.set_page_config(page_title="BTC DCA P/L", page_icon="💰", layout="wide")

ESPLORA_BASE = "https://mempool.space/api"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/price"
UPBIT_TICKER   = "https://api.upbit.com/v1/ticker"

# ---------------- HTTP ----------------
def http_get(url: str, params: Dict | None = None, *, max_retry: int = 3, timeout: int = 8):
    for i in range(max_retry):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(min(5, 2 ** i))
                continue
            r.raise_for_status()
            return r.json()
        except:
            if i < max_retry - 1:
                time.sleep(min(5, 2 ** i))
                continue
            raise
    return None

# ---------------- Price ----------------
@functools.lru_cache(maxsize=4096)
def price_on_date(date: dt.date) -> Decimal:
    start = int(dt.datetime(date.year, date.month, date.day, tzinfo=dt.UTC).timestamp() * 1000)
    end   = int((dt.datetime(date.year, date.month, date.day, tzinfo=dt.UTC) + dt.timedelta(days=1)).timestamp() * 1000)
    data = http_get(BINANCE_KLINES, {
        "symbol": "BTCUSDT", "interval": "1d",
        "startTime": start, "endTime": end, "limit": 1
    })
    return Decimal(str(data[0][4]))

def current_price():
    usd = Decimal(str(http_get(BINANCE_TICKER, {"symbol": "BTCUSDT"})["price"]))
    try:
        krw = Decimal(str(http_get(UPBIT_TICKER, {"markets": "KRW-BTC"})[0]["trade_price"]))
    except:
        krw = None
    return usd, krw

# ---------------- Transactions ----------------
def fetch_all_txs(addr: str):
    """ 페이지네이션 적용: 오래된 거래까지 전부 수집 """
    all_txs, last = [], None
    while True:
        url = f"{ESPLORA_BASE}/address/{addr}/txs/chain" + (f"/{last}" if last else "")
        page = http_get(url, timeout=8)
        if not page:
            break
        all_txs.extend(page)
        if len(page) < 25:
            break
        last = page[-1]["txid"]
    return all_txs

def fetch_wallet_events_parallel(addresses: List[str]):
    addrset = set(addresses)
    tx_map: Dict[str, Dict] = {}
    progress = st.progress(0, text="주소 스캔 중...")
    total = len(addresses)

    def process_addr(addr):
        txs = fetch_all_txs(addr)
        for tx in txs:
            bt = tx.get("status", {}).get("block_time")
            if not bt:
                continue
            t = tx_map.setdefault(tx["txid"], {"time": bt, "recv": Decimal(0), "spent": Decimal(0)})
            for vout in tx.get("vout", []):
                if vout.get("scriptpubkey_address") in addrset:
                    t["recv"] += Decimal(vout.get("value", 0)) / Decimal(1e8)
            for vin in tx.get("vin", []):
                prev = vin.get("prevout", {})
                if prev.get("scriptpubkey_address") in addrset:
                    t["spent"] += Decimal(prev.get("value", 0)) / Decimal(1e8)
        return addr

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 4)) as ex:
        futures = {ex.submit(process_addr, a): a for a in addresses}
        for idx, f in enumerate(as_completed(futures), start=1):
            _ = f.result()
            progress.progress(idx / total, text=f"주소 스캔 중... ({idx}/{total})")

    progress.empty()
    events = []
    for v in tx_map.values():
        net = v["recv"] - v["spent"]
        if net != 0:
            d = dt.datetime.fromtimestamp(v["time"], dt.UTC).date()
            events.append({"date": d, "net_btc": net})
    events.sort(key=lambda e: e["date"])
    return events

# ---------------- FIFO ----------------
def fifo_apply(lots, withdrawals):
    lots = sorted([[d, a] for d, a in lots], key=lambda x: x[0])
    wds  = sorted(withdrawals, key=lambda x: x[0])
    for wd_date, wd_amt in wds:
        remain = wd_amt; i = 0
        while remain > Decimal("1e-12") and i < len(lots):
            use = min(lots[i][1], remain)
            lots[i][1] -= use; remain -= use
            if lots[i][1] <= Decimal("1e-12"):
                lots.pop(i); continue
            i += 1
    return [(d, a) for d, a in lots if a > Decimal("1e-12")]

# ---------------- XPUB ----------------
SLIP_PREFIX_MAP = {
    "xpub": ("p2pkh", 44), "ypub": ("p2sh-p2wpkh", 49),
    "zpub": ("p2wpkh", 84), "vpub": ("p2tr", 86),
}
def detect_prefix(x: str) -> Optional[str]:
    p = x.strip()[:4].lower()
    return p if p in SLIP_PREFIX_MAP else None

def _ctx_from_xpub(xpub: str):
    typ, _ = SLIP_PREFIX_MAP[detect_prefix(xpub)]
    if typ == "p2pkh":   return Bip44.FromExtendedKey(xpub, Bip44Coins.BITCOIN)
    if typ == "p2sh-p2wpkh": return Bip49.FromExtendedKey(xpub, Bip49Coins.BITCOIN)
    if typ == "p2wpkh":  return Bip84.FromExtendedKey(xpub, Bip84Coins.BITCOIN)
    if typ == "p2tr":    return Bip86.FromExtendedKey(xpub, Bip86Coins.BITCOIN)
    raise ValueError("Unsupported XPUB type")

def derive_xpub_addresses(xpub: str, change: int, count: int):
    if not BIP_OK:
        raise RuntimeError("bip-utils 미설치 필요")
    chain = _ctx_from_xpub(xpub).Change(Bip44Changes.CHAIN_EXT if change==0 else Bip44Changes.CHAIN_INT)
    return [chain.AddressIndex(i).PublicKey().ToAddress() for i in range(count)]

# ---------------- Build DF ----------------
def build_rows(addresses):
    usd_now, _ = current_price()
    events = fetch_wallet_events_parallel(addresses)
    deposits = [(ev["date"], ev["net_btc"]) for ev in events if ev["net_btc"] > 0]
    withdrawals = [(ev["date"], -ev["net_btc"]) for ev in events if ev["net_btc"] < 0]
    remaining = fifo_apply(deposits, withdrawals)

    rows = []
    for d, amt in remaining:
        usd_then = price_on_date(d)
        rows.append({
            "date": d.isoformat(),
            "amount_btc": amt,
            "price_usd_at_date": usd_then,
            "current_price_usd": usd_now,
            "cost_usd": amt * usd_then,
            "value_usd": amt * usd_now,
            "pl_usd": amt * (usd_now - usd_then),
        })
    return pd.DataFrame(rows)

# ---------------- UI ----------------
st.title("💰 BTC DCA 수익 계산기 (속도+그래프+XPUB)")
mode = st.radio("입력 방법", ["주소 직접 입력", "XPUB/Descriptor 입력"])

if mode == "주소 직접 입력":
    lines = st.text_area("BTC 주소 (줄바꿈으로 여러 개)")
    if st.button("계산하기"):
        addrs = [a.strip() for a in lines.splitlines() if a.strip()]
        if addrs:
            df = build_rows(addrs)
            if not df.empty:
                s_cost = df["cost_usd"].sum()
                s_val = df["value_usd"].sum()
                pct = (s_val/s_cost - 1) * 100 if s_cost > 0 else 0
                st.metric("총 보유 BTC", f"{df['amount_btc'].sum():.8f}")
                st.metric("총 원가", f"${s_cost:,.2f}")
                st.metric("총 평가액", f"${s_val:,.2f}")
                st.metric("수익률", f"{pct:.2f}%")
                st.dataframe(df)

                # 그래프
                fig = px.line(df, x="date", y=["cost_usd", "value_usd"], title="BTC 가치 추세 (USD)")
                st.plotly_chart(fig, use_container_width=True)

else:
    if not BIP_OK:
        st.error("bip-utils 설치 필요: pip install bip-utils")
    else:
        xpub = st.text_input("XPUB / ypub / zpub / vpub")
        count = st.number_input("파생 주소 개수", 10, 200, 50)
        include_change = st.checkbox("변경체인 포함", value=True)
        if st.button("스캔 및 계산"):
            addrs = derive_xpub_addresses(xpub, change=0, count=count)
            if include_change:
                addrs += derive_xpub_addresses(xpub, change=1, count=count)
            addrs = list(dict.fromkeys(addrs))
            df = build_rows(addrs)
            if not df.empty:
                s_cost = df["cost_usd"].sum()
                s_val = df["value_usd"].sum()
                pct = (s_val/s_cost - 1) * 100 if s_cost > 0 else 0
                st.metric("총 보유 BTC", f"{df['amount_btc'].sum():.8f}")
                st.metric("총 원가", f"${s_cost:,.2f}")
                st.metric("총 평가액", f"${s_val:,.2f}")
                st.metric("수익률", f"{pct:.2f}%")
                st.dataframe(df)

                # 그래프
                fig = px.line(df, x="date", y=["cost_usd", "value_usd"], title="BTC 가치 추세 (USD)")
                st.plotly_chart(fig, use_container_width=True)
