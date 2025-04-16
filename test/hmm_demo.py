import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 直接添加 test 的父目录（aura_source）

from my_models.hmm_model import MyHMM
