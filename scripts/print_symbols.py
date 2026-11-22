import json

symbols = json.load(open("top_symbols.json", encoding="utf-8"))
for s in symbols:
    print(f"InsCode: {s['ins_code']}, Symbol: {repr(s['symbol'])}, Name: {repr(s['name'])}, English: {s.get('english_name')}, Days: {s['days']}")
