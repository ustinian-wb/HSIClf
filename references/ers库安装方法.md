https://github.com/aniku777tw/ERS-EntropyRateSuperpixel/tree/main?tab=readme-ov-file

原文作者提供了在python3.x版本下的源码
但是作者已经生成了pyd文件 这个时候如果运行的话会出现DLL Load failed:找不到指定的模块
这里我尝试了一些措施,比如将要引入的模块直接添加到python路径,但都没有用，需要重新编译


# build

进入目录

```
cd ERS-EntropyRateSuperpixel-main\ERS_Python
```

找到文件`compile_cpp.py`

修改路径为python对应版本

```python
shutil.copy2('build\lib.win-amd64-3.9\ERSModule.cp39-win_amd64.pyd', 'ERSModule.pyd')
```

在conda进入虚拟环境，执行命令：

    python compile_cpp.py build  

会报error，忽略。

进入`build\lib.win-amd64-cpython-39`，找到`ERSModule.cp39-win_amd64.pyd`，更名为`ERSModule.pyd`

进入虚拟环境的`site-packages`，直接放入该`.pyd`文件。



# usage

```python
from ERSModule import *

label_list = ERS(img_list, h, w, nC)
```
