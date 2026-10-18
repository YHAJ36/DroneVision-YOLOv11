"""VisDrone2019-DET 标注格式转换：VisDrone -> YOLO"""
import os
import shutil
from collections import Counter
from pathlib import Path

# 项目根目录：本脚本位于 code/train，向上 2 级即项目根，路径一律基于它拼
ROOT = Path(__file__).resolve().parents[2]

from PIL import Image

#路径配置
SRC_TRAIN = str(ROOT / "dataset_visdrone" / "raw_train" / "VisDrone2019-DET-train")
SRC_VAL = str(ROOT / "dataset_visdrone" / "raw_val" / "VisDrone2019-DET-val")
DST = str(ROOT / "dataset_visdrone" / "yolo_self")

#丢弃类型：0-忽略区，11 = others
DROP_CATEGORIES = {0, 11}

def convert_one_split(split_name,src_root,dst_root):
    """转换一个子集。split_name取'train'或'val'"""
    #步骤1：创建目标目录
    src_img_dir = os.path.join(src_root, "images")
    src_ann_dir = os.path.join(src_root, "annotations")
    dst_img_dir = os.path.join(dst_root, "images", split_name)
    dst_lbl_dir = os.path.join(dst_root, "labels", split_name)

    if not os.path.isdir(src_img_dir) or not os.path.isdir(src_ann_dir):
        print("[!] 源目录缺失，跳过%s"  %  split_name)
        return
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True) 

 
    # 临时自检
    print("  源 images      : %d 个文件" % len(os.listdir(src_img_dir)))
    print("  源 annotations : %d 个文件" % len(os.listdir(src_ann_dir)))
    print("  输出 images    -> %s" % dst_img_dir)
    print("  输出 labels    -> %s" % dst_lbl_dir)
    stats = Counter()
    #步骤2：遍历标注文件
    ann_files = sorted(os.listdir(src_ann_dir))
    print("待处理文件：%d个" % len(ann_files))
    stats["标注文件"] = len(ann_files)

    for idx,ann_name in enumerate(ann_files):
        #拆出文件名主部和扩展名：
        stem,ext = os.path.splitext(ann_name)
        #拼出对应图片的完整路径：
        img_path = os.path.join(src_img_dir, stem + ".jpg")
        #防御：标注有但图片缺失
        if not os.path.exists(img_path):
            print("[!] 图片缺失，跳过：%s" % ann_name)
            continue
        #临时自检，只打印前三个
        if idx < 3:
            print("%s -> %s" % (ann_name, os.path.basename(img_path)))
        #步骤3：读图片真实尺寸
        with Image.open(img_path) as im:
            img_w,img_h = im.size

        if idx < 3:
            print("  图片尺寸：%d x %d" % (img_w,img_h))
        #步骤4：解析标注并换算坐标
        lines_out = []
        ann_path = os.path.join(src_ann_dir, ann_name)
        with open(ann_path,"r",encoding="utf-8") as f:
            for raw_line in f:
                raw_line =raw_line.strip()
                if not raw_line:
                    continue

                parts = raw_line.split(",")
                while parts and parts[-1] == "":
                    parts.pop()

                if len(parts) < 8:
                    continue

                x=float(parts[0])
                y=float(parts[1])
                w=float(parts[2])
                h=float(parts[3])
                category_id=int(float(parts[5]))

                if category_id in DROP_CATEGORIES:
                    continue
                if w <= 0 or h <= 0:
                    continue

                cx = (x + w/2) / img_w
                cy = (y + h/2) / img_h
                nw = w / img_w
                nh = h / img_h

                class_id = category_id - 1
                lines_out.append("%d %.6f %.6f %.6f %.6f" % (class_id, cx, cy, nw, nh))
        #步骤5：写标签+复制图片
        lbl_path = os.path.join(dst_lbl_dir,stem + ".txt")
        with open(lbl_path,"w",encoding="utf-8") as f:
            if lines_out:
                f.write("\n".join(lines_out) + "\n")

        shutil.copy2(img_path, os.path.join(dst_img_dir, stem + ".jpg"))
        stats["输出图片"] += 1
        stats["有效框"] += len(lines_out)
        if not lines_out:
            stats["空标签"] += 1
    #函数结束前打印统计
    print("---统计---")
    print("标注文件：%s" % stats["标注文件"])
    print("输出图片：%s" % stats["输出图片"])
    print("有效框:%s" % stats["有效框"])
    print("空标签:%s" % stats["空标签"])   


def main():
    convert_one_split("train", SRC_TRAIN, DST)
    convert_one_split("val", SRC_VAL, DST)

if __name__ == "__main__":
    main()
