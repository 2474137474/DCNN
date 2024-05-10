from models.MobileNet_0_5 import mobilenet_0_5
from models.MobileNet_1_0 import mobilenet_1_0
from models.mobilenet import mobilenet

# 在init.py中定义可以被外界调用的类和方法
__all__ = ['mobilenet_0_5', 'mobilenet_1_0', 'mobilenet']