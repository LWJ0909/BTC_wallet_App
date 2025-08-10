"""
BTC Cost Basis & Unrealized P/L Calculator by Deposit Date
----------------------------------------------------------
Purpose
  - For each incoming BTC transfer to your address, look up the historical
    BTC price on that date and compute cost basis, current value, and P/L.
  - Aggregate across all deposits to show total % gain like a stock portfolio.

Notes
  - Mode A (default): Deposit-based P/L (ignores later spends). Simple and
    good for DCA wallets that only receive and hold.
  - Mode B (experimental): UTXO-aware P/L (tracks spends with FIFO). TODO stub.

APIs (no API key needed)
  - Address tx history: Blockstream API
      https://blockstream.info/api/address/{address}/txs
  - Price history by date: CoinGecko
      https://api.coingecko.com/api/v3/coins/bitcoin/history?date=DD-MM-YYYY
  - Current price: CoinGecko simple price
      https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd,krw

Usage
  - Set BTC_ADDRESSES below to one or more addresses (legacy, segwit, taproot OK).
  - Run the script. It prints a table and saves results to CSV.

Limitations
  - If you regularly spend from the same address, Mode A will overstate current
    holdings. For full accuracy, use an xpub and implement UTXO tracking (TODO).
  - CoinGecko "history" returns end-of-day price in UTC; you can switch to
    OHLC daily endpoint if you prefer open/close/avg.
"""

from __future__ import annotations
import csv
import dataclasses as dc
import datetime as dt
import functools
import json
import sys
import time
from typing import Dict, List, Tuple

import requests

# ----------------------------- Config ---------------------------------
BTC_ADDRESSES = [
    "bc1qmhuc0zehc88cuvts4epjxvmc459snwkfx0vzh3"
    ]

VS_CURRENCIES = ["usd", "krw"]  # output currencies
MODE = "deposit"  # "deposit" or "utxo" (utxo not implemented yet)
SAVE_CSV_PATH = "btc_pl_by_deposit.csv"
REQUEST_SLEEP = 0.9  # seconds between API calls to be polite
# 내가 소유한 주소들(여러 개 가능) → 내부이체 자동 무시
MY_ADDRESSES = {
    # "bc1q내지갑1", "bc1q내지갑2", ...
}

# 거래소 입금 주소(또는 prefix) 힌트: 여기에 걸리면 'realized'로 간주
EXCHANGE_ADDR_PREFIXES = [
    # 예) "bc1qxyz...", "3J98t1...",  # 실제 쓰는 거래소 예시를 넣어라
]

# 수동 오버라이드: 이 TXID들은 매도로 간주(실현 P/L 계산에 포함)
TXIDS_SOLD = {
    # "txid1...", "txid2..."
}


# ----------------------------- Models ---------------------------------
@dc.dataclass
class Deposit:
    address: str
    txid: str
    block_time: int  # epoch seconds
    amount_btc: float

    @property
    def date(self) -> dt.date:
        return dt.datetime.fromtimestamp(self.block_time, dt.UTC).date()

@dc.dataclass
class Row:
    address: str
    txid: str
    date: dt.date
    amount_btc: float
    price_usd_at_date: float
    price_krw_at_date: float | None
    current_price_usd: float
    current_price_krw: float | None

    @property
    def cost_usd(self) -> float:
        return self.amount_btc * self.price_usd_at_date

    @property
    def value_usd(self) -> float:
        return self.amount_btc * self.current_price_usd

    @property
    def pl_usd(self) -> float:
        return self.value_usd - self.cost_usd

    @property
    def cost_krw(self) -> float | None:
        return None if self.price_krw_at_date is None else self.amount_btc * self.price_krw_at_date

    @property
    def value_krw(self) -> float | None:
        return None if self.current_price_krw is None else self.amount_btc * self.current_price_krw

    @property
    def pl_krw(self) -> float | None:
        if self.cost_krw is None or self.value_krw is None:
            return None
        return self.value_krw - self.cost_krw

# ----------------------------- Helpers --------------------------------
BLKSTREAM = "https://blockstream.info/api"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"          # daily OHLCV
BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/price"    # current price
UPBIT_TICKER   = "https://api.upbit.com/v1/ticker"                 # current KRW price


def http_get(url: str, params: Dict | None = None):
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def fetch_wallet_events(addresses: List[str]):
    """
    지갑(주소 묶음) 단위 순변화 이벤트 생성 + 라벨링.
    - 내부이체: 순변화 0 → 무시
    - 순유출(음수): 목적지 주소가 거래소 힌트/오버라이드면 realized(매도), 아니면 external-unknown
    - 순유입(양수): deposit
    반환: 오래된 순서의 리스트. 각 원소:
      {'date': date, 'net_btc': float, 'txid': str, 'tag': 'deposit'|'withdraw-realized'|'withdraw-external'}
    """
    addrset = set(addresses)
    tx_map = {}  # txid -> {'time': int, 'recv': int, 'spent': int, 'outs': [addr...]}

    for addr in addresses:
        txs = http_get(f"{BLKSTREAM}/address/{addr}/txs")
        for tx in txs:
            status = tx.get("status", {})
            t = tx_map.setdefault(tx["txid"], {"time": status.get("block_time"), "recv": 0, "spent": 0, "outs": []})
            if not t["time"]:
                continue
            # 받은 합 / 출력 주소 수집
            for vout in tx.get("vout", []):
                spk = vout.get("scriptpubkey_address")
                if spk:
                    t["outs"].append(spk)
                if spk in addrset:
                    t["recv"] += int(vout.get("value", 0))
            # 보낸 합
            for vin in tx.get("vin", []):
                prev = vin.get("prevout", {})
                if prev.get("scriptpubkey_address") in addrset:
                    t["spent"] += int(prev.get("value", 0))

    events = []
    for txid, v in tx_map.items():
        if not v["time"]:
            continue
        net_sats = v["recv"] - v["spent"]
        if net_sats == 0:
            continue  # 내부이체
        date = dt.datetime.fromtimestamp(v["time"], dt.UTC).date()

        if net_sats > 0:
            events.append({"date": date, "net_btc": net_sats/1e8, "txid": txid, "tag": "deposit"})
        else:
            # 순유출: realized 조건 체크
            is_override = txid in TXIDS_SOLD
            is_exchange = any(any(out.startswith(pfx) for pfx in EXCHANGE_ADDR_PREFIXES) for out in v["outs"])
            tag = "withdraw-realized" if (is_override or is_exchange) else "withdraw-external"
            events.append({"date": date, "net_btc": net_sats/1e8, "txid": txid, "tag": tag})

    events.sort(key=lambda e: e["date"])
    return events



@functools.lru_cache(maxsize=4096)
def price_on_date(date: dt.date):
    """
    Return (usd, krw_at_date) for the given UTC date.
    - usd: Binance BTCUSDT daily close
    - krw_at_date: None (간단화)  # 필요하면 업비트 일봉으로 확장 가능
    """
    # UTC 00:00 ~ +1day 00:00 밀리초
    start = int(dt.datetime(date.year, date.month, date.day, tzinfo=dt.UTC).timestamp() * 1000)
    end   = int((dt.datetime(date.year, date.month, date.day, tzinfo=dt.UTC) + dt.timedelta(days=1)).timestamp() * 1000)

    data = http_get(BINANCE_KLINES, {
        "symbol": "BTCUSDT",
        "interval": "1d",
        "startTime": start,
        "endTime": end,
        "limit": 1
    })
    if not data:
        raise RuntimeError(f"No kline for {date.isoformat()}")
    close_price = float(data[0][4])  # [4] = close
    return close_price, None  # KRW은 생략(원하면 확장)



def current_price():
    # USD from Binance
    b = http_get(BINANCE_TICKER, {"symbol": "BTCUSDT"})
    usd = float(b["price"])

    # KRW from Upbit (optional)
    try:
        u = http_get(UPBIT_TICKER, {"markets": "KRW-BTC"})
        krw = float(u[0]["trade_price"])
    except Exception:
        krw = None

    return usd, krw

def build_rows(addresses: List[str]) -> List[Row]:
    usd_now, krw_now = current_price()
    events = fetch_wallet_events(addresses)

    # 1) 로트/출금 분리 + 라벨
    deposits: List[Deposit] = []
    wd_realized: List[Tuple[dt.date, float]] = []
    wd_external: List[Tuple[dt.date, float]] = []

    for ev in events:
        if ev["net_btc"] > 0:  # deposit
            deposits.append(Deposit(
                address="(wallet)", txid=ev["txid"],
                block_time=int(dt.datetime(ev["date"].year, ev["date"].month, ev["date"].day, tzinfo=dt.UTC).timestamp()),
                amount_btc=ev["net_btc"]
            ))
        else:  # withdraw
            amt = -ev["net_btc"]
            if ev["tag"] == "withdraw-realized":
                wd_realized.append((ev["date"], amt))
            else:
                wd_external.append((ev["date"], amt))

    # 2) 외부출금 먼저 차감(미실현 감소), 그 다음 realized 차감(실현)
    remaining_lots, _ = fifo_apply(deposits, wd_external)
    remaining_lots, realized_pairs = fifo_apply(remaining_lots, wd_realized)  # realized_pairs: (wd_date, amt, lot_date)

    # 3) 남은 로트 → 미실현 평가 Rows
    rows: List[Row] = []
    for lot in remaining_lots:
        usd_then, krw_then = price_on_date(lot.date)
        rows.append(Row(
            address=lot.address, txid=lot.txid, date=lot.date,
            amount_btc=lot.amount_btc,
            price_usd_at_date=usd_then, price_krw_at_date=krw_then,
            current_price_usd=usd_now, current_price_krw=krw_now
        ))

    # 4) (선택) 실현손익 계산
    realized_pl_usd = 0.0
    for wd_date, amt, lot_date in realized_pairs:
        usd_wd, _ = price_on_date(wd_date)
        usd_lot, _ = price_on_date(lot_date)
        realized_pl_usd += amt * (usd_wd - usd_lot)

    # 요약에 실현손익도 넣을 수 있게 반환에 첨부 (간단히 튜플로 함께 반환)
    # main() 쪽에서 받아서 출력하도록 살짝 수정하면 됨
    build_rows.realized_pl_usd = realized_pl_usd  # 속성으로 붙여서 전달
    return rows



def fifo_apply(deposits: List[Deposit], withdrawals: List[Tuple[dt.date, float]]):
    """
    deposits: Deposit 객체 리스트 (date, amount_btc)
    withdrawals: [(date, amount_btc), ...]
    반환:
      remaining_lots: 출금 차감 후 남은 Deposit 리스트
      realized: 실현 손익 계산용 원시 데이터(옵션) - 여기선 금액만 리턴
    """
    # 날짜 순 정렬
    lots = sorted(deposits, key=lambda d: d.date)
    wds = sorted(withdrawals, key=lambda x: x[0])

    # 단순 FIFO 차감
    realized_fifo = []  # [(withdraw_date, amount_btc, from_lot_date, from_lot_amount_used)]
    for wd_date, wd_amt in wds:
        remain = wd_amt
        i = 0
        while remain > 1e-12 and i < len(lots):
            lot = lots[i]
            use = min(lot.amount_btc, remain)
            if use > 0:
                realized_fifo.append((wd_date, use, lot.date))
                lot.amount_btc -= use
                remain -= use
                if lot.amount_btc <= 1e-12:
                    # lot 소진되면 제거
                    lots.pop(i)
                    continue
            i += 1
        # 남은 출금이 있는데 lot이 부족하면 0 이하가 될 수 있으나, 여기선 무시(음수 방지)
    return lots, realized_fifo


def summarize(rows: List[Row]):
    total_btc = sum(r.amount_btc for r in rows)

    total_cost_usd = sum(r.cost_usd for r in rows)
    total_value_usd = sum(r.value_usd for r in rows)
    pl_usd = total_value_usd - total_cost_usd
    pct_usd = (pl_usd / total_cost_usd * 100.0) if total_cost_usd > 0 else 0.0

    total_cost_krw = sum(r.cost_krw or 0.0 for r in rows)
    total_value_krw = sum(r.value_krw or 0.0 for r in rows)
    pl_krw = total_value_krw - total_cost_krw if total_cost_krw else None
    pct_krw = (pl_krw / total_cost_krw * 100.0) if total_cost_krw else None

    return {
        "total_btc": total_btc,    
        "total_cost_usd": total_cost_usd,
        "total_value_usd": total_value_usd,
        "pl_usd": pl_usd,
        "pct_usd": pct_usd,
        "total_cost_krw": total_cost_krw,
        "total_value_krw": total_value_krw,
        "pl_krw": pl_krw,
        "pct_krw": pct_krw,
    }


def save_csv(rows: List[Row], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "address",
            "txid",
            "date(UTC)",
            "amount_btc",
            "price_usd_at_date",
            "price_krw_at_date",
            "current_price_usd",
            "current_price_krw",
            "cost_usd",
            "value_usd",
            "pl_usd",
            "cost_krw",
            "value_krw",
            "pl_krw",
        ])
        for r in rows:
            w.writerow([
                r.address,
                r.txid,
                r.date.isoformat(),
                f"{r.amount_btc:.8f}",
                f"{r.price_usd_at_date:.2f}",
                f"{r.price_krw_at_date:.0f}" if r.price_krw_at_date is not None else "",
                f"{r.current_price_usd:.2f}",
                f"{r.current_price_krw:.0f}" if r.current_price_krw is not None else "",
                f"{r.cost_usd:.2f}",
                f"{r.value_usd:.2f}",
                f"{r.pl_usd:.2f}",
                f"{(r.cost_krw or 0):.0f}" if r.cost_krw is not None else "",
                f"{(r.value_krw or 0):.0f}" if r.value_krw is not None else "",
                f"{(r.pl_krw or 0):.0f}" if r.pl_krw is not None else "",
            ])


def main():
    if not BTC_ADDRESSES:
        print("[!] Set BTC_ADDRESSES to your address list.")
        sys.exit(1)

    print(f"Fetching deposits for {len(BTC_ADDRESSES)} address(es)...")
    rows = build_rows(BTC_ADDRESSES)
    if not rows:
        print("No confirmed deposits found.")
        return
    
    totals = {}
    for r in rows:
        totals[r.address] = totals.get(r.address, 0) + r.amount_btc
    print("\nPer-address BTC totals:")
    for addr, amt in totals.items():
        print(f"  {addr}: {amt:.8f} BTC")

    s = summarize(rows)
    print(f"\nTotal BTC (by deposits, after withdrawals): {s['total_btc']:.8f} BTC")
    real_pl = getattr(build_rows, "realized_pl_usd", 0.0)
    if abs(real_pl) > 1e-9:
        print(f"Realized P/L (USD): ${real_pl:.2f}")

    print("\n=== Summary (USD) ===")
    print(f"Cost:  ${s['total_cost_usd']:.2f}")
    print(f"Value: ${s['total_value_usd']:.2f}")
    print(f"P/L:   ${s['pl_usd']:.2f} ({s['pct_usd']:.2f}%)")

    if s["total_cost_krw"]:
        print("\n=== Summary (KRW) ===")
        print(f"Cost:  ₩{s['total_cost_krw']:.0f}")
        print(f"Value: ₩{s['total_value_krw']:.0f}")
        print(f"P/L:   ₩{s['pl_krw']:.0f} ({s['pct_krw']:.2f}%)")

    save_csv(rows, SAVE_CSV_PATH)
    print(f"\nSaved details to {SAVE_CSV_PATH}")


if __name__ == "__main__":
    main()
