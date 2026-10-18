"""临时验证脚本：验证配置，不启动脚本训练"""

import importlib.util
import os
import yaml

spec = importlib.util.spec_from_file_location("train",os.path.join(os.path.dirname(__file__),"train.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)#只执行顶层

print("检查结果")

for name in ("DATA","WEIGHTS","PROJECT"):
    p = getattr(mod,name)
    print("%-8s %-52s %s" % (name,p,"存在√"if os.path.exists(p)else"不存在×"))


with open (mod.DATA,"r",encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print("  yaml    nc=%s, names=%s 条, train=%s" % (
    cfg.get("nc"), len(cfg.get("names") or {}), cfg.get("train")))

from ultralytics import YOLO
m = YOLO(mod.WEIGHTS)
print("  模型    task=%s, nc=%s, 参数量=%s" % (m.task, m.model.yaml.get("nc"),format(sum(p.numel() for p in m.model.parameters()), ",")))
print("\n全部通过 —— train.py 配置正确（未启动训练）")