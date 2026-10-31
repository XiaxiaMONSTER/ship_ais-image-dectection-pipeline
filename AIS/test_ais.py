from pyais import decode

with open("2点_ais.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # 去掉时间戳（例如 [123123123123]!AIVDM,...）
        if line.startswith("["):
            idx = line.find("]")
            if idx != -1:
                line = line[idx+1:]

        if line.startswith("!AIVDM") or line.startswith("!AIVDO"):
            try:
                msg = decode(line)
                print(msg.asdict())
            except:
                pass
