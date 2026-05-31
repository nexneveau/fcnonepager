def process_funds_csv(file_bytes: bytes, skip_risk: bool = False) -> bytes:
    # ---------------------------------------------------------
    # ENCODING FIX: Try standard UTF-8, fallback to Excel Latin-1
    # ---------------------------------------------------------
    def safe_read_csv(skip_rows):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), header=skip_rows, dtype=str, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(file_bytes), header=skip_rows, dtype=str, encoding='latin1')

    # Handle the specific UWEALTH title row format
    df = safe_read_csv(0)
    cols = [str(c).strip() for c in df.columns]
    
    if "Fund Name" not in cols:
        df = safe_read_csv(1)
        cols = [str(c).strip() for c in df.columns]
    
    df.columns = cols
    unnamed = [c for c in df.columns if c.startswith("Unnamed:") or c == ""]
    df.drop(columns=unnamed, inplace=True, errors="ignore")
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df = df[df["Fund Name"].notna() & (df["Fund Name"] != "")].reset_index(drop=True)

    universe = load_universe()
    name_index = {f["name"].lower().strip(): f for f in universe if f.get("name")}
    
    results = []
    for _, row in df.iterrows():
        fund_name = str(row.get("Fund Name", "") or "").strip()
        sales_charge = str(row.get("Fund Sales Charge (%)", "") or "").strip() or None
        fund = match_by_name(fund_name, name_index)

        if fund is None:
            row_out = {h: None for h in HEADERS}
            row_out["Fund Name"] = fund_name
            row_out["Fund Sales Charge (%)"] = sales_charge
            results.append(row_out)
            continue

        risk = {"sharpe_3y": None, "stddev_3y": None}
        if not skip_risk:
            risk = fetch_risk_measures(isin=fund.get("isin"), mstar_id=fund.get("id"))
            time.sleep(0.25)

        row_out = {}
        for header, key in OUTPUT_COLUMNS:
            if key == "_sales_charge": row_out[header] = sales_charge
            elif key in ("sharpe_3y", "stddev_3y"): row_out[header] = risk.get(key)
            else: row_out[header] = fund.get(key)
        results.append(row_out)

    # ---------------------------------------------------------
    # Convert back to a CSV byte stream (Using utf-8-sig for Excel)
    # ---------------------------------------------------------
    out_df = pd.DataFrame(results, columns=HEADERS)
    output_stream = io.StringIO()
    out_df.to_csv(output_stream, index=False)
    
    # 'utf-8-sig' adds a BOM, guaranteeing Excel opens the downloaded file flawlessly
    return output_stream.getvalue().encode('utf-8-sig')
