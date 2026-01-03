"""
命令行入口
使用方法: python -m src --model rf --features base,text
"""
from .main import parse_args, train

if __name__ == '__main__':
    args = parse_args()
    train(args)
