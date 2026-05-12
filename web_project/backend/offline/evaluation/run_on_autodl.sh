#!/bin/bash
# AutoDL 训练一键脚本
# 用法: bash run_on_autodl.sh

set -e

cd /root/autodl-tmp/autodl_package

echo "=== 环境检查 ==="
python3 --version
echo "TensorFlow:"
python3 -c "import tensorflow as tf; print(tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print(f'GPU: {gpus}')" 2>/dev/null || echo "  TF 未安装或导入失败"

echo ""
echo "=== 安装 Python 依赖 ==="
pip install numpy pandas scikit-learn tqdm tabulate -q

echo ""
echo "=== 开始训练 YouTubeDNN ==="
echo "参数: emb_dim=32, neg_sample=50, batch_size=512, epochs=20"
python3 autodl_train.py

echo ""
echo "=== 打包结果 ==="
if [ -d saved_models ]; then
    tar czf trained_results.tar.gz saved_models/
    echo "输出文件: /root/autodl-tmp/autodl_package/trained_results.tar.gz"
else
    echo "错误: saved_models/ 目录不存在，训练可能失败"
    exit 1
fi
