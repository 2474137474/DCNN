from models.MobileNet_0_5 import mobilenet_0_5
<<<<<<< HEAD
from models.MobileNet_1_0 import mobilenet_1_0
=======
from models.MobileNet_1_0_modified import mobilenet_1_0
>>>>>>> 086bb67504fca0476fac0854675b1d8768343506
from models.mobilenet import mobilenet

# 在init.py中定义可以被外界调用的类和方法
__all__ = ['mobilenet_0_5', 'mobilenet_1_0', 'mobilenet']